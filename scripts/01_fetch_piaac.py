#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_fetch_piaac.py
=================
Downloads the PIAAC public use files for the United States and verifies, by
execution, the two questions Study 2 left open:

  Q1  Does the file carry an earnings variable granular enough for a median
      split?  (Cycle 1: EARNMTHALLDCL; cycle 2: EARNMTHALLDCLC2. Monthly
      earnings deciles either way.)
  Q2  What sub-national geography does it carry?  (Cycle 1: CTRYRGN with 9
      regions. Cycle 2: none; CTRYRGN is constant and REG_TL2 is suppressed
      by NCES, so a constant CTRYRGN is reported as INFO, not failure.)

Run from the repository root (the OECD host is not reachable from every
network; this script uses resumable range requests):

    python scripts/01_fetch_piaac.py            # cycle 2 (2022-23), primary
    python scripts/01_fetch_piaac.py --cycle1   # also try the cycle 1 US file

Exit codes: 0 verified; 1 a check failed; 2 network blocked everywhere.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, os, sys, time
import urllib.request, urllib.error

BASE2 = "https://webfs.oecd.org/piaac/cy2-puf-data/CSV/"
BASE1 = "https://webfs.oecd.org/piaac/cy1-puf-data/CSV/"
US2 = "prgusap2.csv"           # confirmed present, 35,241,173 bytes (listing of 7 Sep 2026)
US1_CANDIDATES = ["prgusap1_2017.csv", "prgusap1_2012.csv", "prgusap1.csv"]

NETWORK_SIGNS = ("10060", "10061", "10054", "timed out", "Connection reset",
                 "forcibly closed", "actively refused", "Temporary failure",
                 "CERTIFICATE", "403", "407")

EARN_PREFIXES = ("EARN", "D_Q16", "D2_Q16", "D_Q17", "D_Q18", "MONTHLYINC",
                 "YEARLYINC")
GEO_NAMES = ("CTRYRGN", "REGION", "REG_", "GEO", "CNTRYID")
PROXY_NAMES = ("B_Q01A", "B2_Q01", "EDCAT", "YRSQUAL", "EDLEVEL", "AGE_R",
               "AGEG5LFS", "AGEG10LFS", "GENDER_R", "C_D05", "C2_D05",
               "EMPSTAT", "ISCOSKIL4")
SKILL_PREFIXES = ("PVLIT", "PVNUM", "PVAPS")   # plausible values
WEIGHT_PREFIXES = ("SPFWT",)                    # SPFWT0 + replicate weights

EXPECTED_BYTES = {"prgusap2.csv": 35_241_173}   # OECD listing of 7 Sep 2026


def sniff_delimiter(path: str) -> str:
    """Cycle-1 PUF CSVs are comma-delimited, cycle-2 semicolon-delimited."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def remote_size(url: str, timeout: int = 60):
    """Content-Length from a HEAD request, or None when unavailable."""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "study2-verify/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:                                            # noqa: BLE001
        return None


def looks_blocked(e: Exception) -> bool:
    s = repr(e)
    return any(m in s for m in NETWORK_SIGNS)


def download_resumable(url: str, target: str, tries: int = 6, timeout: int = 120) -> None:
    """Resume with HTTP Range on every retry; verify final size against the
    server. A file already matching the server's Content-Length is not
    re-requested (a Range at end-of-file draws HTTP 416, which is also
    treated as completion rather than an error)."""
    expected = remote_size(url)
    for attempt in range(1, tries + 1):
        have = os.path.getsize(target) if os.path.exists(target) else 0
        if have and expected and have == expected:
            print(f"    already complete ({have:,} bytes, equals server "
                  "Content-Length); download skipped")
            return
        req = urllib.request.Request(url, headers={"User-Agent": "study2-verify/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(target, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            final = os.path.getsize(target)
            if expected and final != expected:
                raise IOError(f"downloaded {final:,} bytes but the server "
                              f"reports {expected:,}")
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                print(f"    server refused the resume range at {have:,} bytes "
                      "(HTTP 416): file complete")
                return
            raise
        except Exception as e:                                   # noqa: BLE001
            if attempt == tries or not looks_blocked(e):
                raise
            wait = min(30, 2 ** attempt)
            print(f"    retry {attempt}/{tries} after {wait}s: {e!r}")
            time.sleep(wait)


def inventory(path: str, label: str) -> dict:
    """Read the header and profile the columns the study depends on."""
    delim = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header = next(csv.reader(f, delimiter=delim))
    up = [h.strip().upper() for h in header]
    found = {
        "earnings": [h for h in up if any(h.startswith(p) for p in EARN_PREFIXES)],
        "geography": [h for h in up if any(g in h for g in GEO_NAMES)],
        "proxies":   [h for h in up if any(h.startswith(p) for p in PROXY_NAMES)],
        "skills":    [h for h in up if any(h.startswith(p) for p in SKILL_PREFIXES)],
        "weights":   [h for h in up if any(h.startswith(p) for p in WEIGHT_PREFIXES)],
    }
    print(f"\n  [{label}] {len(up)} columns, delimiter '{delim}'")
    for k, v in found.items():
        print(f"    {k:9s}: {len(v):3d}  {', '.join(sorted(v)[:12])}{' ...' if len(v) > 12 else ''}")
    return {"header": up, "delimiter": delim, **found}


def profile_key_vars(path: str, inv: dict, label: str) -> bool:
    """Row counts, missingness and category counts for the gating variables."""
    import pandas as pd  # local import so the fetch works without pandas
    want = []
    for name in ("EARNMTHALLDCL", "EARNMTHALLDCLC2", "EARNHRDCL", "EARNHRDCLC2",
                 "EARNHRBONUSDCL", "EARNHRBONUSDCLC2", "EARNFLAGC2", "CTRYRGN",
                 "GENDER_R", "AGE_R", "B_Q01A", "B_Q01A_T", "B2_Q01", "EDCAT8",
                 "EDCAT8_TC1", "EDCAT7", "YRSQUAL", "YRSQUALC2", "EDLEVEL3",
                 "SPFWT0", "VEMETHOD", "VEMETHODN", "VENREPS", "VEFAYFAC"):
        if name in inv["header"]:
            want.append(name)
    pvs = [c for c in inv["header"] if c.startswith(("PVLIT", "PVNUM"))][:2]
    usecols = want + pvs
    df = pd.read_csv(path, sep=inv.get("delimiter", ","),
                     usecols=lambda c: c.strip().upper() in usecols,
                     low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    n = len(df)
    print(f"\n  [{label}] rows: {n:,}")
    ok = True
    for c in sorted(df.columns):
        col = pd.to_numeric(df[c], errors="coerce")
        nonmiss = int(col.notna().sum())
        cats = int(col.nunique())
        print(f"    {c:16s} non-missing {nonmiss:6,} ({100*nonmiss/max(n,1):5.1f}%)  distinct {cats}")

    earn = next((g for g in ("EARNMTHALLDCLC2", "EARNMTHALLDCL")
                 if g in df.columns), None)
    if earn is None:
        print("    !! EARNMTHALLDCL(C2) ABSENT: the written contingency applies")
        ok = False
    else:
        col = pd.to_numeric(df[earn], errors="coerce")
        cats = int(col.nunique()); share = col.notna().mean()
        verdict = "PASS" if cats >= 8 and share > 0.30 else "CHECK"
        suffix = " (cycle 2 name of EARNMTHALLDCL)" if earn.endswith("C2") else ""
        print(f"    {verdict}: {earn}{suffix} has {cats} categories, "
              f"{100*share:.1f}% populated")
        ok = ok and (verdict == "PASS")

    if "CTRYRGN" not in df.columns:
        print("    !! CTRYRGN ABSENT: the written contingency applies")
        ok = False
    else:
        col = pd.to_numeric(df["CTRYRGN"], errors="coerce")
        cats = int(col.nunique()); share = col.notna().mean()
        if cats >= 2 and share > 0.30:
            print(f"    PASS: CTRYRGN has {cats} categories, {100*share:.1f}% populated")
        elif cats == 1:
            print("    INFO: CTRYRGN is constant; the US cycle-2 PUF carries no "
                  "sub-national geography (NCES suppressed the TL2 region: the "
                  "US sample is not regionally representative). Decision of "
                  "2026-09-07: demographic strata replace region strata for "
                  "cycle 2; regional kappa is a cycle-1 sensitivity layer only.")
        else:
            print(f"    CHECK: CTRYRGN has {cats} categories, {100*share:.1f}% populated")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle1", action="store_true", help="also fetch the cycle 1 US file")
    ap.add_argument("--outdir", default="piaac_data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    blocked_everywhere = True
    all_ok = True

    jobs = [(BASE2 + US2, os.path.join(a.outdir, US2), "cycle 2 USA")]
    if a.cycle1:
        jobs += [(BASE1 + c, os.path.join(a.outdir, c), f"cycle 1 USA ({c})")
                 for c in US1_CANDIDATES]

    for url, target, label in jobs:
        print(f"\n== {label}: {url}")
        try:
            download_resumable(url, target)
            blocked_everywhere = False
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    404, not present under this name; skipping")
                continue
            print(f"    HTTP {e.code}: {e}")
            all_ok = False
            continue
        except Exception as e:                                   # noqa: BLE001
            print(f"    unreachable: {e!r}")
            all_ok = False
            continue
        size = os.path.getsize(target)
        sha = hashlib.sha256(open(target, "rb").read()).hexdigest()
        print(f"    {size:,} bytes  sha256 {sha}")
        listed = EXPECTED_BYTES.get(os.path.basename(target))
        if listed:
            print(f"    size check against the listed {listed:,} bytes: "
                  f"{'MATCH' if size == listed else 'MISMATCH'}")
            if size != listed:
                all_ok = False
        inv = inventory(target, label)
        all_ok = profile_key_vars(target, inv, label) and all_ok
        blocked_everywhere = False

    if blocked_everywhere:
        print("\nRESULT: network blocked everywhere (exit 2)")
        return 2
    print(f"\nRESULT: {'ALL CHECKS PASS' if all_ok else 'AT LEAST ONE CHECK FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
