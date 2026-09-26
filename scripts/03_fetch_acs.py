#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the ACS person files for the grid, resumably, keeping the archives.

    python scripts/03_fetch_acs.py --root acs_data              # priority years
    python scripts/03_fetch_acs.py --root acs_data --full       # all years
    python scripts/03_fetch_acs.py --root acs_data --states WY CO CA --years 2022
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from study2lib.acs import fetch_acs_cell, STATE_FIPS
from study2lib.specs import PRIORITY_YEARS, FULL_YEARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="acs_data")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--states", nargs="*", default=sorted(STATE_FIPS))
    ap.add_argument("--years", nargs="*", type=int, default=None)
    a = ap.parse_args()
    years = a.years or (FULL_YEARS if a.full else PRIORITY_YEARS)
    jobs = [(s, y) for y in years for s in a.states]
    print(f"{len(jobs)} state-year files -> {a.root}")
    fail = []
    for i, (s, y) in enumerate(jobs, 1):
        try:
            p = fetch_acs_cell(a.root, s, y)
            print(f"  [{i}/{len(jobs)}] {s} {y}  ok  {os.path.getsize(p):,} bytes")
        except Exception as e:
            print(f"  [{i}/{len(jobs)}] {s} {y}  FAILED  {e!r}")
            fail.append((s, y))
    if fail:
        print(f"\n{len(fail)} failed; rerun this script, it resumes: {fail[:10]}")
        return 1
    print("\nall files present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
