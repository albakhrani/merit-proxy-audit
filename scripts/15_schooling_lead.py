#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Between-sex difference in mean schooling per market, in years and in
attainment-code units, for the premarket and the extended samples
(22 Sep 2026, added with the review arms R4 to R12; the quantity the
direction rule turns on).

The manuscript compares women's lead in mean schooling in the ACS with the
sex difference in the mean of the proxy error estimated in PIAAC
(results/piaac_validity.json, in years of qualification). The grid cells do
not store group means, so this script reads each market once and writes
results/schooling_lead.json with, per market and sample: weighted means of
years-coded schooling, of the raw attainment code and of age by sex, the
female weight share and the sample sizes. Nothing is estimated.

    python scripts/15_schooling_lead.py --root acs_data --workers 3

--age2565 (23 Sep 2026) restricts the sample to ages 25 to 65, the sample
of the earners-anchor arms, and writes results/schooling_lead_age2565.json.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _one(args):
    import numpy as np
    from study2lib import specs
    from study2lib.acs import load_cell
    root, state, year, age_range = args
    out = {}
    for cs_name, cols in specs.COVARIATE_SETS.items():
        rec = {}
        for coding in ("years", "code"):
            cell = load_cell(root, state, year, cols, specs.OUTCOME_DEFS[0], schooling=coding,
                             age_range=age_range)
            X, names, g, w = cell["X"], cell["names"], cell["g"], cell["w"]
            j = names.index("SCHL"); i = names.index("AGEP")
            f, m = g == 1, g == 0
            mf = float(np.average(X[f, j], weights=w[f])); mm = float(np.average(X[m, j], weights=w[m]))
            rec[coding] = {"female_mean": mf, "male_mean": mm, "lead_women_minus_men": mf - mm}
            if coding == "years":
                rec["age"] = {"female_mean": float(np.average(X[f, i], weights=w[f])),
                              "male_mean": float(np.average(X[m, i], weights=w[m]))}
                rec["n"] = int(cell["n"]); rec["n_female"] = int(f.sum()); rec["n_male"] = int(m.sum())
                rec["female_weight_share"] = float(w[f].sum() / w.sum())
        out[cs_name] = rec
    return state, year, out


def main():
    import numpy as np
    from study2lib.acs import STATE_FIPS
    from study2lib import specs
    from study2lib.runlog import write_json, config_hash
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None,
                    help="default results/schooling_lead.json, or "
                         "results/schooling_lead_age2565.json with --age2565")
    ap.add_argument("--age2565", action="store_true",
                    help="ages 25 to 65 (the sample of the earners-anchor arms)")
    a = ap.parse_args()
    age_range = (25, 65) if a.age2565 else (18, None)
    a.out = a.out or ("results/schooling_lead_age2565.json" if a.age2565 else "results/schooling_lead.json")
    years = a.years or specs.PRIORITY_YEARS
    jobs = [(a.root, s, y, age_range) for y in years for s in a.states]
    markets, fail = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(_one, j): j for j in jobs}
        for k, f in enumerate(as_completed(futures), 1):
            _, s, y, _ = futures[f]
            try:
                st, yr, rec = f.result()
                markets[f"{st} {yr}"] = rec
                print(f"  [{k}/{len(jobs)}] {st} {yr}  premarket lead {rec['premarket']['years']['lead_women_minus_men']:+.3f} y, "
                      f"extended lead {rec['extended']['years']['lead_women_minus_men']:+.3f} y")
            except Exception as e:
                fail.append((s, y)); print(f"  [{k}/{len(jobs)}] {s} {y}  FAILED  {e!r}")
    summary = {}
    for cs in specs.COVARIATE_SETS:
        for coding in ("years", "code"):
            v = np.array([m[cs][coding]["lead_women_minus_men"] for m in markets.values()])
            summary[f"{cs}_{coding}"] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                         "min": float(v.min()), "max": float(v.max()),
                                         "n_women_lead": int((v > 0).sum()), "n": int(len(v))}
    write_json(a.out, {"markets": markets, "summary": summary, "crosswalk": specs.SCHL_TO_YEARS,
                       "age_range": list(age_range),
                       "config_hash": config_hash({"years": years, "states": a.states,
                                                   **({"age_range": list(age_range)} if a.age2565 else {})})})
    for k, s in summary.items():
        print(f"{k:18s} women's lead: mean {s['mean']:+.3f} median {s['median']:+.3f} "
              f"range [{s['min']:+.3f}, {s['max']:+.3f}]; women lead in {s['n_women_lead']}/{s['n']}")
    print(f"wrote {a.out}" + (f"; {len(fail)} failed, rerun" if fail else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
