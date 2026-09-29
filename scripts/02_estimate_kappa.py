#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate the reliability of the schooling proxy against assessed skill,
overall and by stratum, from a PIAAC US public use file.

    python scripts/02_estimate_kappa.py --file piaac_data/prgusap2.csv --proxy YRSQUALC2
    python scripts/02_estimate_kappa.py --file piaac_data/prgusap1.csv --proxy YRSQUAL \
        --out results/kappa_cycle1.json

Added 28 Sep 2026 (REGISTERED_CHANGES R15): --construct composite adds a
"composite" block whose plausible values are the mean of the literacy and
numeracy values with the same index (piaac.add_composite_pvs); the literacy
and numeracy blocks are written as before:

    python scripts/02_estimate_kappa.py --file piaac_data/prgusap2.csv --proxy YRSQUALC2         --construct composite --out results/kappa_composite.json

Writes results/kappa.json by default. Proxy names resolve across cycles
(YRSQUAL finds YRSQUALC2 and the reverse); if the proxy is absent under
either name the script lists the education variables it can see and exits
rather than guessing. Strata whose columns are constant in the file are
skipped with a printed reason: the US cycle-2 PUF carries no sub-national
geography (NCES suppressed the TL2 region because the US sample is not
regionally representative), so the region strata resolve only on cycle-1
files. Downstream consumers use "overall" (grid, pilot, H2.1) and "sex"
(H2.2); region strata are a descriptive sensitivity layer only.
"""
import argparse, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from study2lib.piaac import (load_piaac, kappa_by_stratum, kappa_sex_difference,
                             resolve_column, add_composite_pvs, PV_LIT, PV_NUM, PV_COMPOSITE)
from study2lib.runlog import write_json

STRATA = {
    "sex": ["GENDER_R"],
    "region": ["CTRYRGN"],
    "age10": ["AGEG10LFS"],
    "educ": ["EDCAT8"],
    "sex_region": ["GENDER_R", "CTRYRGN"],
    "sex_educ": ["GENDER_R", "EDCAT8"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--proxy", default="YRSQUAL")
    ap.add_argument("--out", default="results/kappa.json")
    ap.add_argument("--subset", default="none", choices=["none", "workers"],
                    help="workers: EARNFLAGC2 == 1 and AGEG10LFS bands 2-5 "
                         "(earners aged 25-65), the composition-matched "
                         "anchor added 15 Sep 2026 (post-results, disclosed)")
    ap.add_argument("--construct", default=None, choices=["composite"],
                    help="composite: also estimate the reliability for the mean of the "
                         "literacy and numeracy plausible values (R15)")
    a = ap.parse_args()
    df = load_piaac(a.file)
    if a.construct == "composite":
        df = add_composite_pvs(df)
    print(f"loaded {len(df):,} rows, {len(df.columns)} columns")
    if a.subset == "workers":
        if "EARNFLAGC2" not in df.columns or "AGEG10LFS" not in df.columns:
            print("subset 'workers' needs EARNFLAGC2 and AGEG10LFS; "
                  "absent in this file")
            return 1
        flag = pd.to_numeric(df["EARNFLAGC2"], errors="coerce")
        band = pd.to_numeric(df["AGEG10LFS"], errors="coerce")
        keep = ((flag == 1) & (band >= 2) & (band <= 5)).fillna(False)
        df = df[keep]
        print(f"subset workers (EARNFLAGC2 == 1, AGEG10LFS 2-5): "
              f"{len(df):,} rows")

    proxy = resolve_column(df, a.proxy)
    if proxy is None:
        ed = [c for c in df.columns if c.startswith(
            ("YRSQ", "EDCAT", "B_Q01", "B2_Q01", "EDLEVEL"))]
        print(f"proxy {a.proxy} not in file; education variables present: {ed}")
        print("rerun with --proxy <one of these>")
        return 1
    if proxy != a.proxy:
        print(f"proxy {a.proxy} resolved to its cycle name {proxy}")

    strata, skipped = {}, {}
    for name, cols in STRATA.items():
        resolved = [resolve_column(df, c) for c in cols]
        if any(r is None for r in resolved):
            skipped[name] = "column absent"
            continue
        if proxy in resolved:
            skipped[name] = f"stratifies by the proxy itself ({proxy}); " \
                            "the proxy is constant inside every cell"
            continue
        constant = [r for r in resolved if df[r].nunique(dropna=True) <= 1]
        if constant:
            skipped[name] = f"constant column {constant} (no sub-national " \
                            "geography in the US cycle-2 PUF)" \
                if "CTRYRGN" in constant else f"constant column {constant}"
            continue
        strata[name] = resolved
    for name, why in skipped.items():
        print(f"stratum '{name}' skipped: {why}")

    payload = {"file": os.path.basename(a.file), "proxy": proxy,
               "proxy_requested": a.proxy, "subset": a.subset,
               "strata_skipped": skipped,
               "lit": kappa_by_stratum(df, proxy, PV_LIT, strata),
               "num": kappa_by_stratum(df, proxy, PV_NUM, strata)}
    if a.construct == "composite":
        payload["construct"] = "composite: plausible value k is the mean of PVLIT k and PVNUM k"
        payload["composite"] = kappa_by_stratum(df, proxy, PV_COMPOSITE, strata)
    sexcol = resolve_column(df, "GENDER_R")
    if sexcol and df[sexcol].nunique(dropna=True) > 1:
        payload["lit"]["sex_difference"] = kappa_sex_difference(
            df, proxy, PV_LIT, sex_col=sexcol)
        payload["num"]["sex_difference"] = kappa_sex_difference(
            df, proxy, PV_NUM, sex_col=sexcol)
        if a.construct == "composite":
            payload["composite"]["sex_difference"] = kappa_sex_difference(
                df, proxy, PV_COMPOSITE, sex_col=sexcol)
    write_json(a.out, payload)
    k = payload["lit"]["overall"]
    if k["kappa"] is None:
        print(f"\noverall kappa not estimable: {k['note']}")
        return 1
    print(f"\nkappa(proxy={proxy}, literacy) overall = {k['kappa']:.4f}"
          f" (rho {k['rho']:.4f}, se {k['se']:.4f}, n {k['n']:,},"
          f" {k['replication']}, factor {k['factor']:.5f},"
          f" {k['n_replicates']} replicates)")
    k = payload["num"]["overall"]
    print(f"kappa(proxy={proxy}, numeracy) overall = {k['kappa']:.4f}"
          f" (rho {k['rho']:.4f}, se {k['se']:.4f})")
    if a.construct == "composite":
        k = payload["composite"]["overall"]
        print(f"kappa(proxy={proxy}, composite) overall = {k['kappa']:.4f}"
              f" (rho {k['rho']:.4f}, se {k['se']:.4f})")
        for label, key in (("men", "1"), ("women", "2")):
            ks = payload["composite"].get("sex", {}).get(key)
            if ks and ks.get("kappa") is not None:
                print(f"kappa(composite, {label}) = {ks['kappa']:.4f} (se {ks['se']:.4f}, n {ks['n']:,})")
    for skill in (("lit", "num", "composite") if a.construct == "composite" else ("lit", "num")):
        d = payload[skill].get("sex_difference")
        if d and d.get("difference") is not None:
            print(f"sex difference ({skill}, {d['convention']}): "
                  f"{d['difference']:+.4f} (replicate se {d['se']:.4f}, "
                  f"z {d['z']:.2f})")
    if payload["lit"]["overall"]["kappa"] > 1 or payload["lit"]["overall"]["kappa"] < 0:
        print("!! kappa outside [0, 1]: refuse to use this file downstream")
        return 1
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
