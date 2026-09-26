#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rerun the study end to end from the public sources.

    python reproduce.py [--root acs_data] [--workers 3] [--force] [--list]

This script only calls the existing scripts, in the order and with the
arguments used for the recorded runs. It reimplements nothing. Every step is
resumable: rerunning this command after an interruption continues where the
scripts left off, because each cell script skips cells whose result file
already exists with the current configuration hash.

The script refuses to run while results/ already holds run logs, so that the
recorded logs cannot be overwritten by accident. Pass --force to run into an
existing results/ folder (cells with a matching configuration hash are then
kept and only missing cells are computed).

Approximate cost, from the timing_s fields of the recorded run logs: the
registered grid about 25 hours of cell time, the harmonised arm about 10
hours, the six review arms and the eight mean-error arms together about 40
hours, and the three bootstraps several hours each; with three workers the
recorded runs took several days of wall time on one laptop with an NVIDIA
RTX 4080 Laptop GPU. The two data fetches transfer about 15 GB (ACS) and
35 MB (PIAAC).
"""
import argparse, glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# Sex differences in the mean proxy error, in years, as reported in
# results/piaac_validity.json (all adults: literacy point 0.297 with
# interval 0.159 to 0.435, numeracy point 0.560 with upper bound 0.694;
# earners aged 25 to 65: literacy 0.580, numeracy 0.871). They are passed as
# literals because they are the arguments of the recorded runs.
MEANERROR_ARMS = [
    # (tag, delta, extra arguments)
    ("lit",         "0.297", []),
    ("lit_lo",      "0.159", []),
    ("lit_hi",      "0.435", []),
    ("num",         "0.560", []),
    ("num_hi",      "0.694", ["--decomp-only"]),
    ("lit_workers", "0.580", ["--age2565", "--decomp-only"]),
    ("num_workers", "0.871", ["--age2565", "--decomp-only"]),
]


def steps(root, workers):
    w = ["--workers", str(workers)]

    def s(name):
        return os.path.join("scripts", name)

    piaac = os.path.join("piaac_data", "prgusap2.csv")
    kappa = os.path.join("results", "kappa.json")
    yield "fetch the PIAAC Cycle 2 United States file", [s("01_fetch_piaac.py"), "--outdir", "piaac_data"]
    yield "fetch the ACS one-year PUMS files (2017, 2018, 2019, 2021, 2022; fifty states)", [s("03_fetch_acs.py"), "--root", root]
    yield "reliability anchor, all adults (results/kappa.json)", [s("02_estimate_kappa.py"), "--file", piaac, "--proxy", "YRSQUALC2"]
    yield "reliability anchor, earners aged 25 to 65 (results/kappa_workers.json)", [s("02_estimate_kappa.py"), "--file", piaac, "--proxy", "YRSQUALC2", "--subset", "workers", "--out", os.path.join("results", "kappa_workers.json")]
    yield "reliability anchor, attainment-code proxy (results/kappa_edcat8.json)", [s("02_estimate_kappa.py"), "--file", piaac, "--proxy", "EDCAT8_TC1", "--out", os.path.join("results", "kappa_edcat8.json")]
    yield "pilot timing on three cells (results/pilot.json)", [s("00_pilot.py"), "--root", root, "--kappa", kappa]
    yield "registered grid (results/grid)", [s("04_run_grid.py"), "--root", root, "--kappa", kappa] + w
    yield "harmonised premarket arm (results/grid_h)", [s("06_harmonised.py"), "--root", root, "--kappa", kappa] + w
    yield "bootstrap v1 on the primary arm (results/boot_primary)", [s("07_boot_primary.py"), "--root", root, "--kappa", kappa] + w
    yield "exact arm (results/grid_exact)", [s("12_run_arms.py"), "--arm", "exact", "--root", root] + w
    yield "check that the exact arm reproduces the registered grid", [s("13_check_exact.py")]
    yield "PIAAC differential-validity test (results/piaac_validity.json)", [s("11_piaac_validity.py"), "--file", piaac]
    yield "years-coded schooling arm (results/grid_years)", [s("12_run_arms.py"), "--arm", "years", "--root", root] + w
    for tag, delta, extra in MEANERROR_ARMS:
        outdir = os.path.join("results", "grid_meanerror_" + tag)
        yield "mean-error arm, delta " + delta + " (" + outdir + ")", [s("12_run_arms.py"), "--arm", "meanerror", "--delta", delta, "--outdir", outdir, "--root", root] + extra + w
    yield "years-coded arm on ages 25 to 65 (results/grid_years_age2565)", [s("12_run_arms.py"), "--arm", "meanerror", "--delta", "0", "--age2565", "--decomp-only", "--outdir", os.path.join("results", "grid_years_age2565"), "--root", root] + w
    yield "mean-error summary, all adults (results/meanerror_summary.json)", [s("14_meanerror_summary.py"), "--results", "results", "--tags", "lit", "lit_lo", "lit_hi", "num", "num_hi"]
    yield "mean-error summary, ages 25 to 65 (results/meanerror_summary_workers.json)", [s("14_meanerror_summary.py"), "--results", "results", "--years-dir", os.path.join("results", "grid_years_age2565"), "--tags", "lit_workers", "num_workers", "--out", os.path.join("results", "meanerror_summary_workers.json")]
    yield "bootstrap v2 on the primary arm (results/boot_primary_v2)", [s("07_boot_primary.py"), "--root", root, "--kappa", kappa, "--folds-by-person", "--kappa-draw", "--outdir", os.path.join("results", "boot_primary_v2")] + w
    yield "bootstrap v2 on the mean-error arm (results/boot_meanerror_lit)", [s("07_boot_primary.py"), "--root", root, "--kappa", kappa, "--folds-by-person", "--kappa-draw", "--schooling", "years", "--shift-female-years", "0.297", "--shift-se", "0.069", "--no-grid-check", "--outdir", os.path.join("results", "boot_meanerror_lit")] + w
    for arm in ["wage", "occ", "adjinc", "wmedian", "age2565"]:
        yield arm + " arm (results/grid_" + arm + ")", [s("12_run_arms.py"), "--arm", arm, "--root", root] + w
    yield "schooling lead per market, ages 18 and over (results/schooling_lead.json)", [s("15_schooling_lead.py"), "--root", root] + w
    yield "schooling lead per market, ages 25 to 65 (results/schooling_lead_age2565.json)", [s("15_schooling_lead.py"), "--root", root, "--age2565"] + w
    yield "pooled cross-market estimates (results/pooled_gap.json)", [s("10_pooled_gap.py")]
    yield "registered hypotheses (results/hypotheses.json)", [s("05_hypotheses.py")]


def main():
    ap = argparse.ArgumentParser(description="Rerun the study end to end from the public sources.")
    ap.add_argument("--root", default="acs_data", help="folder for the ACS files (about 15 GB)")
    ap.add_argument("--workers", type=int, default=3, help="worker processes for the cell runs")
    ap.add_argument("--force", action="store_true", help="run even though results/ already holds run logs")
    ap.add_argument("--list", action="store_true", help="print the commands and exit without running anything")
    a = ap.parse_args()
    os.chdir(ROOT)
    plan = list(steps(a.root, a.workers))
    if a.list:
        for i, (what, cmd) in enumerate(plan, 1):
            print("[%2d/%d] %s\n        %s %s" % (i, len(plan), what, PY, " ".join(cmd)))
        return 0
    existing = glob.glob(os.path.join("results", "**", "*.json"), recursive=True)
    if existing and not a.force:
        print("results/ already holds %d run logs; refusing to run. Move them aside, or pass "
              "--force to run into the existing folder (cells with a matching configuration "
              "hash are kept)." % len(existing))
        return 1
    for i, (what, cmd) in enumerate(plan, 1):
        print("\n[%2d/%d] %s\n        %s %s" % (i, len(plan), what, PY, " ".join(cmd)), flush=True)
        r = subprocess.run([PY] + cmd)
        if r.returncode != 0:
            print("step %d exited with code %d; fix the cause and rerun this command "
                  "(completed cells are kept)" % (i, r.returncode))
            return r.returncode
    print("\nall steps completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
