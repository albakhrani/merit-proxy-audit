# Correcting a merit proxy for measurement error does not close the sex gap in income audits

Code and run logs for the manuscript of that title, submitted to Scientific Reports.

Authors: Saleh Abdul Amir Mohammad (School of Economics and Management, Dalian University of Technology; ORCID 0009-0001-7758-4513), Xin Ye (School of Economics and Management, Dalian University of Technology), Qian Chen (School of Economics and Management, Dalian University of Technology) and Ali A. AL-Bakhrani (corresponding author; Faculty of Administrative Sciences and Computing, Albaydha University, Yemen, and School of Software, Dalian University of Technology; ORCID 0000-0003-1360-8640).

## What the study does

The study asks whether the sex gap that an income audit attributes to unequal treatment survives once the audit's merit proxy, years of schooling, is corrected for measurement error. It estimates the reliability of schooling as a proxy for assessed skill from the PIAAC Cycle 2 United States file, then runs a registered grid of twofold Oaxaca-Blinder decompositions with an errors-in-variables correction, a reliability sweep and three screening learners across 250 state-year markets of the American Community Survey (fifty states; 2017, 2018, 2019, 2021 and 2022). In every market the corrected residual keeps the sign of the uncorrected one, and the post-registration arms recorded in the amendment ledger (a harmonised sample, years-coded schooling, wages of full-time workers, occupation groups, real dollars, a weighted threshold, an age-restricted anchor, a test of differential validity and a mean-error correction) do not change that conclusion.

## Repository layout

| Path | Contents |
| --- | --- |
| `study2lib/` | The library: cell loading (`acs.py`), the registered specification grid and the amendment ledger (`specs.py`), the decomposition, the errors-in-variables correction and the reliability estimator (`eiv.py`), the PIAAC reliability and validity estimators (`piaac.py`), the learners, the bootstrap, the per-cell runner (`cellrun.py`) and the run-log writer (`runlog.py`). |
| `scripts/` | The numbered steps of the pipeline, listed under "Rerunning the study" below. |
| `tests/` | The offline test suite, on synthetic data with known answers. |
| `results/` | The run logs: one JSON file per state-year cell for the registered grid and every arm, the bootstrap output and the JSON summaries. Every number in the manuscript is read from these files. |
| `paper/` | The scripts that rebuild the manuscript's figure 5, the supplementary tables and the number checks from `results/`. |
| `reproduce.py` | One command that reruns the study end to end from the public sources. |
| `requirements.txt` | The runtime versions recorded in the run logs. |
| `requirements-dev.txt` | The test runner, which the recorded runs did not use. |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata. |
| `LICENSE`, `results/LICENSE` | MIT for the code, CC BY 4.0 for the run logs. |

The code and the amendment ledger were kept in dated snapshots rather than a version-control system before this release, so the ledger dates in `study2lib/specs.py` and the `written_at` timestamps in the run logs are the record of the order in which the grid was fixed, estimated and amended.

## Environment and pinned versions

The recorded runs used Python 3.12.10 on Windows 11 with the versions below, which every run log records in its `versions` field.

| Package | Version |
| --- | --- |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| xgboost | 3.4.1 |
| tabpfn | 8.5.0 |
| torch | 2.14.0+cu130 |
| folktables | 0.0.12 |

PyTorch 2.14.0 with CUDA 13.0 and the TabPFN v2 weights are what the recorded runs used. The `tabpfn` learner pins the v2 weights through `TABPFN_MODEL_VERSION` in `study2lib/specs.py`; the `tabpfn` package downloads them from Hugging Face the first time the learner is called, so the machine needs access to that host, or a mirror named in the `HF_ENDPOINT` environment variable. The CUDA build of PyTorch comes from the PyTorch package index, which the first line of `requirements.txt` names; on a machine without a CUDA 13 GPU the learners run on the CPU.

```
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Fetching the data

The raw inputs are not redistributed. Two fetchers obtain them from their original servers; both resume interrupted downloads.

- `scripts/01_fetch_piaac.py` downloads the PIAAC Cycle 2 United States public-use file `prgusap2.csv` (about 35 MB) from `https://webfs.oecd.org/piaac/cy2-puf-data/CSV/` into `piaac_data/` and checks the columns the study relies on.
- `scripts/03_fetch_acs.py` downloads the ACS one-year PUMS person files for 2017, 2018, 2019, 2021 and 2022 for all fifty states (about 15 GB) from `https://www2.census.gov/programs-surveys/acs/data/pums/` into `acs_data/`.

```
venv\Scripts\python.exe scripts\01_fetch_piaac.py
venv\Scripts\python.exe scripts\03_fetch_acs.py --root acs_data
```

## Running the offline tests

The tests need no data; they run on synthetic files with known answers.

```
venv\Scripts\python.exe -m pytest tests -q
```

Expected: 32 passed. `tests/test_note1.py` prints each figure it checks. Two named tests: the scale invariance and bounds of the reliability estimator (`test_scale_invariance_and_bounds` in `tests/test_piaac_synth.py`) and the validity estimator on synthetic data with a delete-one-group replicate design (`tests/test_validity.py`). Each test file also runs directly as a script, for example `venv\Scripts\python.exe tests\test_core.py`.

## Rerunning the study

One command reruns everything, from the two data fetches to the hypotheses file, by calling the scripts below in order with the arguments of the recorded runs:

```
venv\Scripts\python.exe reproduce.py --root acs_data --workers 3
```

`reproduce.py --list` prints the 48 commands without running them. The script refuses to run while `results/` already holds run logs; move them aside, or pass `--force` to run into the existing folder (cells whose configuration hash matches are kept). Approximate cost, from the `timing_s` fields of the run logs: the registered grid about 25 hours of cell time, the harmonised arm about 10 hours, the review arms and the mean-error arms together about 40 hours, and the three bootstraps several hours each. With three workers the recorded runs took several days of wall time on one laptop with an NVIDIA RTX 4080 Laptop GPU. The worker count affects runtime only: every cell draws its random streams from the primary seed inside the cell (the bootstrap adds a per-cell offset derived from the state code and year), so no result depends on which worker runs a cell or on how many run at once.

The steps, in the order the recorded runs used:

1. `scripts/01_fetch_piaac.py` and `scripts/03_fetch_acs.py`: the data.
2. `scripts/02_estimate_kappa.py`: the reliability anchors (`results/kappa.json`, `kappa_workers.json`, `kappa_edcat8.json`).
3. `scripts/00_pilot.py`: timing on three cells (`results/pilot.json`).
4. `scripts/04_run_grid.py`: the registered grid (`results/grid`).
5. `scripts/06_harmonised.py`: the harmonised premarket arm (`results/grid_h`).
6. `scripts/07_boot_primary.py`: the bootstrap on the primary arm, version 1 (`results/boot_primary`).
7. `scripts/12_run_arms.py --arm exact` and `scripts/13_check_exact.py`: the guard that the library reproduces the registered grid bit for bit (`results/grid_exact`).
8. `scripts/11_piaac_validity.py`: the differential-validity test (`results/piaac_validity.json`).
9. `scripts/12_run_arms.py`: the years-coded arm, the mean-error arms at the estimated deltas, and the wage, occupation, real-dollar, weighted-median and age-restricted arms (`results/grid_years`, `grid_meanerror_*`, `grid_years_age2565`, `grid_wage`, `grid_occ`, `grid_adjinc`, `grid_wmedian`, `grid_age2565`).
10. `scripts/14_meanerror_summary.py`: the mean-error summaries (`results/meanerror_summary.json`, `meanerror_summary_workers.json`).
11. `scripts/07_boot_primary.py --folds-by-person --kappa-draw`: the bootstrap, version 2, on the primary arm and on the mean-error arm (`results/boot_primary_v2`, `boot_meanerror_lit`).
12. `scripts/15_schooling_lead.py`: the between-sex difference in mean schooling per market (`results/schooling_lead.json`, `schooling_lead_age2565.json`).
13. `scripts/10_pooled_gap.py` and `scripts/05_hypotheses.py`: the pooled estimates and the registered hypotheses (`results/pooled_gap.json`, `results/hypotheses.json`).
14. `scripts/02_estimate_kappa.py --construct composite` and `scripts/11_piaac_validity.py --constructs composite`: the reliability anchors and the differential-validity test for the composite construct, whose plausible values are the mean of the literacy and numeracy values (`results/kappa_composite.json`, `kappa_composite_workers.json`, `piaac_validity_composite.json`).
15. `scripts/12_run_arms.py --arm meanerror --share`: the proportional mean-error arms, which lower women's years-coded schooling by a share of each market's own schooling lead, on the full sample and on ages 25 to 65 (`results/grid_prop_lit`, `grid_prop_num`, `grid_prop_lit_workers`, `grid_prop_num_workers`).
16. `scripts/12_run_arms.py --arm meanerror --skill composite`: the mean-error arm at the composite delta and the composite anchor, with the literacy anchor stored as well (`results/grid_meanerror_composite`).
17. `scripts/14_meanerror_summary.py`: the summaries of the proportional and composite arms (`results/meanerror_summary_prop.json`, `meanerror_summary_prop_workers.json`, `meanerror_summary_composite.json`).
18. `scripts/16_replicate_variance.py`: successive-difference replication standard errors of the primary decomposition from the ACS replicate weights, compared with the bootstrap standard errors (`results/replicate_se.json`). Replicate weights that are negative in the source file (0.0011 per cent of person-replicate entries) are set to zero in this script and counted per market in its output.
19. `scripts/17_groupref.py`: the decomposition under the male and under the female coefficients as reference, uncorrected and corrected at the common and at the sex-specific anchors, with its summary (`results/grid_groupref`, `results/groupref_summary.json`).
20. `scripts/18_dummies.py`: the pooled-reference decomposition with the attainment code entered as indicator columns, uncorrected, with its summary (`results/grid_dummies`, `results/dummies_summary.json`).

## The registered plan and the amendment ledger

`study2lib/specs.py` holds the registered specification grid: the outcome definitions, the covariate sets, the learners, the reliability sweep, the seed, the bootstrap size and the powered-cell rule. Any change after the first real cell was estimated is an entry in `REGISTERED_CHANGES` in the same file, with its date, the change and the reason. The ledger has 19 entries: the first, dated 14 September 2026, was made before any real cell was estimated and lowers the floor of the reliability sweep; the other 18, dated 15, 22, 25 and 28 September 2026, are flagged `post_results` and add the arms and corrections that the manuscript reports (entries R14 to R18, dated 28 September 2026, are the review arms listed as steps 14 to 20 above).

## Configuration hashes

Every cell file under `results/` carries `config_hash` (a SHA-256 prefix of the configuration that produced it), `seed`, `versions` and, where a learner was fitted, `tabpfn_model_version`. Every file in a folder carries the same hash, and each hash recomputes from `study2lib/specs.py` and the reliability files alone:

| Folder or file | config_hash |
| --- | --- |
| `grid` | `de6c1429664b99d0` |
| `grid_h` | `18e962f1ccbe1170` |
| `grid_exact` | `051841f8abfc0e81` |
| `boot_primary` | `381fcfc6c3e96093` |
| `boot_primary_v2` | `207ea10fe71c683e` |
| `boot_meanerror_lit` | `5c9147c33cd3bc9d` |
| `grid_years` | `bd87b125e0656a62` |
| `grid_years_age2565` | `d2ffd47464f73216` |
| `grid_meanerror_lit` | `696315867a0fb27d` |
| `grid_meanerror_lit_lo` | `62b98f53749cc871` |
| `grid_meanerror_lit_hi` | `d25e6007c78fbd4c` |
| `grid_meanerror_num` | `4d4e07fd6952aa0b` |
| `grid_meanerror_num_hi` | `f2afe68fe306a1e0` |
| `grid_meanerror_lit_workers` | `76bce16d4df8fd0a` |
| `grid_meanerror_num_workers` | `54548c91e55c45c7` |
| `grid_wage` | `680f55416a1118f0` |
| `grid_adjinc` | `fa7a5c62a177dba6` |
| `grid_wmedian` | `1d746efccb6d6e83` |
| `grid_age2565` | `519f45228d8398d3` |
| `grid_occ` | `824e73d366ff38da` |
| `grid_prop_lit` | `79e45b5dea64cacb` |
| `grid_prop_num` | `a6fbd018d2f494d3` |
| `grid_prop_lit_workers` | `35d102211e490fe4` |
| `grid_prop_num_workers` | `d6dfbee7fd11705d` |
| `grid_meanerror_composite` | `694c31dcd98afa38` |
| `grid_groupref` | `bbba22a80306930d` |
| `grid_dummies` | `3785b4b9739b0054` |
| `pooled_gap.json` | `b8550a8e79548b18` |
| `piaac_validity.json` | `b1621d3a0603a87e` |
| `schooling_lead.json` | `0676c828685f79e1` |
| `schooling_lead_age2565.json` | `46dda144c3bd9b01` |

## Rebuilding the manuscript's numbers

The scripts in `paper/` read the run logs under `results/` only and write nothing into that folder. Run them from the repository root; each defaults to `results/` for the run logs and to `paper/` for its outputs.

- `paper/verify_numbers.py [results_dir]` checks every number quoted in the main text against the run logs and ends with a count of checks and failures (197 checks, 0 failed on the released logs).
- `paper/build_si_tables.py [results_dir] [out_dir]` writes the supplementary tables `paper/tables/S1.tex` to `S10B.tex` and `paper/si_numbers.json`, every scalar the Supplementary Note quotes. The released `tables/` and `si_numbers.json` are the copies used for the submitted manuscript; a rebuild reproduces the tables exactly and the JSON to within floating-point rounding (largest relative difference below 1e-12).
- `paper/build_si.py` writes `paper/si.tex`, the Supplementary Note with every number substituted from `paper/si_numbers.json` and the tables.
- `paper/fig5_meanerror.py [results_dir] [out_stem]` draws figure 5 from the mean-error summaries (needs matplotlib, listed in `requirements-dev.txt`).

## Licences

The code is under the MIT licence (`LICENSE`). The run logs under `results/` are under the Creative Commons Attribution 4.0 International licence (`results/LICENSE`).

## Citation

Mohammad, S. A. A., Ye, X., Chen, Q. and AL-Bakhrani, A. A. (2026). Code and run logs for: Correcting a merit proxy for measurement error does not close the sex gap in income audits (version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22986776

The concept DOI https://doi.org/10.5281/zenodo.22986775 always resolves to the latest version.

Machine-readable metadata are in `CITATION.cff`.
