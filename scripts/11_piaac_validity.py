#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differential validity of the schooling proxy by sex in PIAAC
(REGISTERED_CHANGES R10, 22 Sep 2026, post-results; simulated referee
report item M2).

The manuscript names a sex difference in the mean of the proxy error as the
one violation of the classical model that could reverse the direction of
the correction, and says the assessment survey can test it. This script
runs that test: assessed skill regressed on schooling and a female
indicator, with plausible values combined by Rubin's rules and replicate
weights for the variance, against the classical-error benchmark for the
female coefficient (which is not zero: women's lead in mean schooling alone
produces b_X (1 - kappa)/kappa times that lead). See
study2lib.piaac.differential_validity for the algebra.

    python scripts/11_piaac_validity.py --file piaac_data/prgusap2.csv

Runs, for literacy and numeracy, in the full adult sample and in the
earners-aged-25-to-65 subset used for the second anchor:
    years_noage   YRSQUALC2 linear, no age terms (the clean test)
    years_age     YRSQUALC2 linear plus age (AGE_R linear if present with
                  enough distinct values, otherwise AGEG10LFS band dummies)
    edcat8_age    EDCAT8_TC1 category dummies plus age (female coefficient
                  only; no benchmark is defined for a categorical proxy)

Added 28 Sep 2026 (REGISTERED_CHANGES R15): --constructs selects the
constructs to run; "composite" uses plausible values that are the mean of
the literacy and numeracy values with the same index (piaac.add_composite_pvs):

    python scripts/11_piaac_validity.py --file piaac_data/prgusap2.csv         --constructs composite --out results/piaac_validity_composite.json

Writes results/piaac_validity.json and prints one line per specification.
Reading the output: "difference" is the female coefficient minus the
classical benchmark, with its replicate-based standard error; "delta" is
the implied sex difference in the mean of the proxy error in years of
qualification, positive where women's schooling overstates their assessed
skill relative to men's (the case that could shrink the corrected gap).
"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from study2lib.piaac import (load_piaac, resolve_column, differential_validity,
                             add_composite_pvs, pv_columns, PV_LIT, PV_NUM)
from study2lib.runlog import write_json, config_hash
from study2lib import specs


def age_terms(df):
    """AGE_R linear when the public file carries it with enough distinct
    values; otherwise AGEG10LFS band dummies (drop first). Returns
    (matrix or None, description)."""
    if "AGE_R" in df.columns:
        a = pd.to_numeric(df["AGE_R"], errors="coerce")
        if a.notna().sum() > 0.9 * len(df) and a.nunique() >= 10:
            return a.to_numpy(float), "AGE_R linear"
    if "AGEG10LFS" in df.columns:
        b = pd.to_numeric(df["AGEG10LFS"], errors="coerce")
        cats = sorted(b.dropna().unique())
        if len(cats) >= 2:
            mat = np.column_stack([(b == c).astype(float).to_numpy() for c in cats[1:]])
            mat[b.isna().to_numpy()] = np.nan
            return mat, f"AGEG10LFS bands, {len(cats)} categories, first dropped"
    return None, "no age variable found"


def fmt(s, nd=3):
    if s is None:
        return "   n/a"
    return f"{s['point']:+.{nd}f} ({s['se']:.{nd}f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--proxy", default="YRSQUAL")
    ap.add_argument("--out", default="results/piaac_validity.json")
    ap.add_argument("--constructs", nargs="*", default=["lit", "num"],
                    choices=["lit", "num", "composite"],
                    help="constructs to run; composite is the mean of literacy and numeracy (R15)")
    a = ap.parse_args()
    df = load_piaac(a.file, extra_cols=("EDCAT8_TC1", "EDCAT8", "EARNFLAGC2", "AGEG10LFS"))
    if "composite" in a.constructs:
        df = add_composite_pvs(df)
    proxy = resolve_column(df, a.proxy)
    if proxy is None:
        print(f"proxy {a.proxy} not in file")
        return 1
    edcat = resolve_column(df, "EDCAT8")
    sexcol = resolve_column(df, "GENDER_R")
    print(f"loaded {len(df):,} rows; proxy {proxy}; attainment coding {edcat}; sex {sexcol}")

    samples = {"all": np.ones(len(df), bool)}
    if "EARNFLAGC2" in df.columns and "AGEG10LFS" in df.columns:
        flag = pd.to_numeric(df["EARNFLAGC2"], errors="coerce")
        band = pd.to_numeric(df["AGEG10LFS"], errors="coerce")
        samples["workers"] = ((flag == 1) & (band >= 2) & (band <= 5)).fillna(False).to_numpy()
    else:
        print("EARNFLAGC2 or AGEG10LFS absent: the workers subset is skipped")

    out = {"file": os.path.basename(a.file), "proxy": proxy, "attainment_coding": edcat,
           "convention": "female coefficient at given schooling (and age); benchmark is "
                         "the classical-error value b_X (1 - kappa_within)/kappa_within * "
                         "(women's lead in mean schooling); difference = coefficient - "
                         "benchmark; delta = -difference * kappa_within / b_X, in proxy "
                         "units, positive where women's schooling overstates skill",
           "specs": {}}
    header = (f"{'skill':4s} {'sample':8s} {'spec':12s} {'n':>6s} {'b_female':>17s} "
              f"{'benchmark':>17s} {'difference':>17s} {'z':>6s} {'delta(yrs)':>17s} "
              f"{'kappa_w':>7s} {'b_x':>8s}")
    print(header)
    if "composite" in a.constructs:
        out["construct"] = "composite: plausible value k is the mean of PVLIT k and PVNUM k"
    for skill, pv in ((c, pv_columns(c)) for c in a.constructs):
        out["specs"][skill] = {}
        for sname, mask in samples.items():
            sub = df[mask]
            agemat, agedesc = age_terms(sub)
            block = {"age_terms_used": agedesc, "n_rows": int(len(sub))}
            block["years_noage"] = differential_validity(sub, proxy, pv, sex_col=sexcol)
            block["years_age"] = (differential_validity(sub, proxy, pv, sex_col=sexcol, age=agemat)
                                  if agemat is not None else None)
            block["edcat8_age"] = (differential_validity(sub, edcat, pv, sex_col=sexcol,
                                                         age=agemat, proxy_dummies=True)
                                   if (edcat and agemat is not None) else None)
            out["specs"][skill][sname] = block
            for spec in ("years_noage", "years_age", "edcat8_age"):
                r = block[spec]
                if r is None:
                    continue
                z = r["difference"]["z"] if r["difference"] else None
                ztxt = "%+.2f" % z if z is not None else "n/a"
                ktxt = "%.3f" % r["kappa_within"]["point"] if r["kappa_within"] else "n/a"
                btxt = "%.3f" % r["b_x"]["point"] if r["b_x"] else "n/a"
                print("%-4s %-8s %-12s %6d %17s %17s %17s %6s %17s %7s %8s"
                      % (skill, sname, spec, r["n"], fmt(r["b_female"]), fmt(r["benchmark"]),
                         fmt(r["difference"]), ztxt, fmt(r["delta_proxy_units"]), ktxt, btxt))
    out["config_hash"] = config_hash({"proxy": proxy, "edcat": edcat, "samples": list(samples),
                                      "seed": specs.PRIMARY_SEED,
                                      **({"constructs": a.constructs} if a.constructs != ["lit", "num"] else {})})
    write_json(a.out, out)
    head = a.constructs[0]
    r = out["specs"][head]["all"]["years_noage"]
    print(f"\nheadline ({head}, all adults, no age terms): difference "
          f"{r['difference']['point']:+.3f} (se {r['difference']['se']:.3f}, z "
          f"{r['difference']['z']:+.2f}); delta {r['delta_proxy_units']['point']:+.3f} years "
          f"(se {r['delta_proxy_units']['se']:.3f})")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
