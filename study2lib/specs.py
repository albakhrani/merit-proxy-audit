# -*- coding: utf-8 -*-
"""The registered specification grid. Immutable once estimation begins.

Any change after the first real cell is estimated must be recorded in
REGISTERED_CHANGES with a date and a reason, and reported in the paper.
"""

PRIMARY_SEED = 20261001

# Analyst degrees of freedom swept by the specification curve (H2.5)
OUTCOME_DEFS = ["acs50k", "cellmedian"]        # income > $50,000 (folktables ACSIncome) vs above cell median
COVARIATE_SETS = {
    "premarket": ["AGEP", "SCHL"],             # Study 1's leakage rule: information available before the outcome
    "extended":  ["AGEP", "SCHL", "WKHP", "COW"],  # adds hours and class of worker; OCCP excluded (dimensionality; recorded)
}
LEARNERS = ["logistic", "xgboost", "tabpfn"]   # three model generations
KAPPA_MODES = ["uncorrected", "corrected_point"]

# The kappa sweep reported per cell (H2.1, and the flip point).
# Registered 7 Sep 2026 as 0.50..1.00; floor lowered 14 Sep 2026, see
# REGISTERED_CHANGES[0].
KAPPA_GRID = [round(0.15 + 0.05 * i, 2) for i in range(18)]  # 0.15 .. 1.00

# Proxy measured with error (kappa applies); demographics treated as error-free
PROXY_COLUMN = "SCHL"

# TabPFN weights generation used by the 'tabpfn' learner. tabpfn >= 8 resolves
# its default to TabPFN-3, whose weights sit behind a Prior Labs licence token;
# the v2 weights are open and match the 10,000-row context cap in learners.py.
# Switching to "v3" needs TABPFN_TOKEN set and, once any real cell exists, a
# REGISTERED_CHANGES entry. Recorded in every cell result and in the grid hash.
TABPFN_MODEL_VERSION = "v2"

# Group contrast for the main grid
GROUP = "SEX"                                   # ACS: 1 male, 2 female; gap reported female minus male

# Grid extent
PRIORITY_YEARS = [2017, 2018, 2019, 2021, 2022]  # 2020 excluded: standard 1-Year ACS not released
FULL_YEARS = [2013, 2014, 2015, 2016] + PRIORITY_YEARS + [2023]

# Powered-cell rule (declared minimum cell size, per group)
MIN_GROUP_N = 200

# Bootstrap: primary arms only, to bound compute; the spec curve itself is point estimates
N_BOOT = 500
BOOT_ARMS = [("acs50k", "premarket", "logistic")]  # intervals on the primary arm; other arms are point estimates in the curve

# Harmonised arm (post-results addition R1, 15 Sep 2026; see REGISTERED_CHANGES):
# premarket covariates estimated on the extended set's complete-case sample,
# so the covset contrast decomposes into sample and conditioning components.
HARMONISED_SAMPLE_REQUIRES = ["WKHP", "COW"]

# Concordance (H2.5): share of powered cells where >= 90% of available core
# specifications agree with the modal verdict; prediction: below 0.75
GAMMA_AGREE = 0.90
GAMMA_PREDICTED_BELOW = 0.75
OMEGA_TOLERANCE = 0.05                          # auditor tolerance on R, absolute
OMEGA_PREDICTED_BELOW = 0.50

# ---- Arms added after the internal review of 22 Sep 2026 (post-results;
# REGISTERED_CHANGES R4-R12) ----
# ACS attainment code SCHL (1-24) to years of schooling. Codes 1-3 (no
# schooling, nursery, kindergarten) carry 0; 4-14 are grades 1-11; 15 is
# grade 12 without a diploma; 16-17 diploma or GED; 18-19 some college;
# 20 associate; 21 bachelor; 22 master; 23 professional; 24 doctorate.
# The two contestable cells are 23 and 24; the values chosen are written
# here and nowhere else.
SCHL_TO_YEARS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
                 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 12, 17: 12, 18: 13,
                 19: 14, 20: 14, 21: 16, 22: 18, 23: 19, 24: 21}

# OCCP four-digit codes to SOC major groups, by code range (inclusive).
# Ranges follow the Census occupation code list; the military and the
# "unemployed, last worked 5+ years ago" codes form one residual group.
OCCP_MAJOR_GROUPS = [
    (10, 440, "management"), (500, 750, "business"), (800, 960, "finance"),
    (1005, 1240, "computer_math"), (1305, 1560, "architecture_engineering"),
    (1600, 1980, "science"), (2001, 2060, "community_social"),
    (2100, 2180, "legal"), (2205, 2555, "education"), (2600, 2970, "arts_media"),
    (3000, 3550, "health_practitioners"), (3601, 3655, "health_support"),
    (3700, 3960, "protective"), (4000, 4160, "food"), (4200, 4255, "cleaning"),
    (4330, 4655, "personal_care"), (4700, 4965, "sales"), (5000, 5940, "office"),
    (6005, 6130, "farming"), (6200, 6950, "construction"),
    (7000, 7640, "installation_repair"), (7700, 8990, "production"),
    (9005, 9760, "transportation"), (9800, 9999, "military_or_none"),
]

# Arm definitions consumed by scripts/12_run_arms.py. Every key maps to
# load_cell keyword options; absent keys take the registered defaults, so
# an empty dict reproduces the registered grid.
ARMS = {
    "exact":   {},                                    # registered grid, logistic only, exact anchors stored
    "years":   {"schooling": "years"},
    "wage":    {"income_var": "WAGP", "fulltime": True},
    "occ":     {"occupation": True},
    "adjinc":  {"adjinc": True},
    "wmedian": {"weighted_median_threshold": True},
    "age2565": {"age_range": (25, 65)},               # run with the earners anchor (kappa_workers.json)
}

REGISTERED_CHANGES = [
    {"date": "2026-09-14",
     "change": "KAPPA_GRID floor lowered from 0.50 to 0.15, step 0.05 unchanged",
     "reason": "the PIAAC cycle-2 anchor (results/kappa.json, proxy YRSQUALC2: "
               "kappa 0.1867 literacy se 0.0178, 0.2264 numeracy se 0.0169; "
               "proxy EDCAT8_TC1 within 0.008 of the same values) lies below "
               "the registered floor, so the sweep as registered never passed "
               "through the estimated reliability and every flip point below "
               "0.50 would have been an extrapolation. Change made before any "
               "real grid cell was estimated; eiv_correct's positive-"
               "definiteness guard records any cell not correctable at low "
               "kappa instead of reporting coefficients from an indefinite "
               "system."},
    {"date": "2026-09-15", "post_results": True,
     "change": "R1: harmonised premarket arm added (premarket covariates on "
               "the WKHP/COW-complete sample; scripts/06_harmonised.py, "
               "results/grid_h)",
     "reason": "adversarial review found the covset contrast confounded "
               "with a sample change (the extended set drops WKHP/COW-"
               "missing rows, median 70 percent of the premarket n) and the "
               "load_cell docstring's ACSIncome-mirroring claim false (the "
               "kit filters AGEP>=18 and PINCP>0 only). The harmonised arm "
               "decomposes the premarket-to-extended verdict flip into "
               "sample and conditioning components. Existing grid outputs "
               "are untouched; defaults reproduce them bit for bit."},
    {"date": "2026-09-15", "post_results": True,
     "change": "R2: joint weighted bootstrap on the primary arm "
               "(scripts/07_boot_primary.py, results/boot_primary): pps-"
               "within-sex draws, intervals for delta_label, delta_merit, R, "
               "the uncorrected and corrected residuals and the deepening; "
               "positive-definiteness failures counted per cell; group "
               "moments exported for the H2.2 flip thresholds",
     "reason": "H2.1's registered falsifier referenced an interval for the "
               "corrected residual that no script computed (the T6 "
               "bootstrap covered R only), and the verdict statistic "
               "delta_merit carried no uncertainty anywhere. Correction of "
               "the record: entry 1's claim that a positive-definiteness "
               "failure would be recorded per cell described a mechanism "
               "the grid code did not have; in fact no failure occurred "
               "(all 250 cells carry full 18-point sweeps, so S_XX - "
               "Lambda stayed positive definite everywhere), and this "
               "script now performs the promised recording per draw."},
    {"date": "2026-09-15", "post_results": True,
     "change": "R3: worker-matched PIAAC anchor (subset workers: EARNFLAGC2 "
               "== 1 and AGEG10LFS bands 2-5) and a replicate-based SE for "
               "the sex difference in kappa (kappa_sex_difference); "
               "constant-proxy guard threshold 1e-20",
     "reason": "the pooled 16-65 anchor is dragged down by the 16-24 "
               "student band (kappa 0.026 against 0.21-0.30 in older "
               "bands), which is anti-conservative here because the "
               "explained component is positive in all 250 cells, so a "
               "lower kappa enlarges the correction in the study's own "
               "direction; the sex-difference SE previously used an "
               "independent-subgroup approximation although both sexes "
               "share one Fay design; eight strata cells carried 1e-30 "
               "rounding dust instead of the not-estimable marker."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R4: years-coded schooling arm (SCHL_TO_YEARS crosswalk; "
               "scripts/12_run_arms.py --arm years, results/grid_years)",
     "reason": "simulated referee report of 22 September 2026, item M5.3: "
               "kappa is estimated on years of qualification but applied to "
               "the attainment code entered linearly, whose codes 16-24 are "
               "credentials rather than years; a squared correlation is not "
               "invariant to that recoding. The registered grid is untouched."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R5: wage-and-salary, full-time full-year arm (WAGP > 0, "
               "WKHP >= 35, WKW == 1 or WKWN >= 50; --arm wage, results/grid_wage)",
     "reason": "item M4: annual total personal income with no hours "
               "restriction mixes labour supply with pay, the standard "
               "object in the gender-gap literature being the wage of "
               "full-time workers."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R6: occupation-groups arm (extended set plus OCCP mapped to "
               "SOC major groups, OCCP_MAJOR_GROUPS; --arm occ, results/grid_occ)",
     "reason": "item M4: occupation was excluded at registration on "
               "dimensionality grounds although it is the largest explained "
               "component in most gender decompositions; the major-group "
               "coding keeps the dimension at about two dozen columns."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R7: ADJINC real-dollar arm (income times ADJINC/1e6 before "
               "the 50,000 threshold; --arm adjinc, results/grid_adjinc)",
     "reason": "minor item 7: the registered threshold is nominal across "
               "years. The log-income decomposition is invariant to a "
               "year-constant factor, so this arm bears on R and the strict "
               "verdict only."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R8: PWGTP-weighted cell median for the cellmedian outcome "
               "(--arm wmedian, results/grid_wmedian)",
     "reason": "minor item 6: the registered threshold used the unweighted "
               "median while every other estimate is weighted, recorded as "
               "an inconsistency; this arm repairs it without touching the "
               "registered outputs."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R9: ACS aged 25 to 65 arm propagated at the earners anchor "
               "(kappa_workers.json; --arm age2565, results/grid_age2565), "
               "and the corrected residual at the exact anchors 0.187 and "
               "0.243 stored in every S-batch cell (eiv_unexplained_at)",
     "reason": "item M5, last paragraph: the propagated anchor is national "
               "for ages 16-65 while the screening sample begins at 18 with "
               "no upper bound; the age-restricted anchor was reported only "
               "at the nearest swept value 0.25."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R10: PIAAC differential-validity test (scripts/11_piaac_validity.py, "
               "results/piaac_validity.json): assessed skill on schooling and a "
               "female indicator, with the classical-error benchmark for the "
               "female coefficient computed inside every replicate",
     "reason": "item M2: the manuscript names a sex difference in the mean "
               "of the proxy error as the one violation that could reverse "
               "the direction of the correction and leaves it untested; the "
               "assessment survey can test it."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R11: mean-error arm (women's schooling shifted by the "
               "estimated delta before the correction; --arm meanerror, "
               "results/grid_meanerror), run only if R10 rejects the classical null",
     "reason": "item M2, consequence: the corrected gap under the estimated "
               "error structure, alongside the classical one."},
    {"date": "2026-09-22", "post_results": True,
     "change": "R12: bootstrap v2 (scripts/07_boot_primary.py --folds-by-person "
               "--kappa-draw, results/boot_primary_v2): folds assigned by "
               "original person so duplicates never straddle training and "
               "evaluation, and kappa drawn per draw from its replicate-based "
               "normal, truncated to the sweep range",
     "reason": "item M6: resampling before the out-of-fold refit leaked "
               "duplicates across folds, and the intervals carried no "
               "first-stage uncertainty. v1 outputs are kept untouched."},
    {"date": "2026-09-25", "post_results": True,
     "change": "R13: run_cell records eiv_min_kappa (the smallest reliability for "
               "which S_XX - Lambda stays positive definite, equal to the weighted "
               "R-squared of the proxy on the other covariates) and eiv_correctable, "
               "and stores None for corrected quantities below that threshold instead "
               "of raising; the uncorrected decomposition, the screening statistic and "
               "the admissible part of the reliability sweep are kept",
     "reason": "the occupation arm (R6, item M4) is not correctable at kappa = 0.187 "
               "in any of the 250 markets: occupation explains more of schooling's "
               "variance than the assumed reliability leaves as error. Recording the "
               "threshold reports that fact rather than losing the arm."},
    {"date": "2026-09-28", "post_results": True,
     "change": "R14: proportional mean-error arms (load_cell shift_female_share; "
               "scripts/12_run_arms.py --arm meanerror --share, results/grid_prop_lit, "
               "grid_prop_num, grid_prop_lit_workers, grid_prop_num_workers): women's "
               "years-coded schooling lowered by a share of the cell's own person-weighted "
               "schooling lead instead of a fixed number of years, the share being the "
               "PIAAC estimate delta divided by the PIAAC schooling lead for the matching "
               "sample and construct; decomposition fields only",
     "reason": "pre-submission review of 28 September 2026: the error difference was "
               "transported as a fixed number of years although the schooling lead "
               "differs between the two surveys. Added post-results; every existing "
               "mean-error output is untouched."},
    {"date": "2026-09-28", "post_results": True,
     "change": "R15: composite construct (piaac.add_composite_pvs: plausible value k is "
               "the mean of PVLIT k and PVNUM k; scripts/02_estimate_kappa.py --construct "
               "composite, results/kappa_composite.json and kappa_composite_workers.json; "
               "scripts/11_piaac_validity.py --constructs composite, "
               "results/piaac_validity_composite.json) and one decomposition-only mean-error "
               "arm at the composite delta and the composite anchor, with the literacy "
               "anchor stored through extra_kappas (results/grid_meanerror_composite)",
     "reason": "pre-submission review of 28 September 2026: the construct was a single "
               "skill. Added post-results; the literacy and numeracy anchors and every "
               "existing arm are untouched."},
    {"date": "2026-09-28", "post_results": True,
     "change": "R16: successive-difference replication standard errors for the primary "
               "decomposition from the ACS replicate weights PWGTP1 to PWGTP80 "
               "(scripts/16_replicate_variance.py, results/replicate_se.json), compared "
               "with the bootstrap standard errors implied by results/boot_primary",
     "reason": "pre-submission review of 28 September 2026: the bootstrap ignored the ACS "
               "replicate design. Added post-results; the replicate variance covers "
               "sampling variance of the design only and does not propagate the anchor's "
               "uncertainty, which the version 2 bootstrap does."},
    {"date": "2026-09-28", "post_results": True,
     "change": "R17: group-specific reference structure (decomposition.twofold_groupref; "
               "scripts/17_groupref.py, results/grid_groupref, results/groupref_summary.json): "
               "the outcome equation fitted separately by sex, the twofold decomposition "
               "under the male and under the female coefficients as reference, uncorrected "
               "and corrected at the common literacy anchor and at the sex-specific anchors "
               "of results/kappa.json; premarket covariates, full sample, registered coding",
     "reason": "pre-submission review of 28 September 2026: the differential-reliability "
               "channel was closed only by the pooled-reference estimator. Added "
               "post-results; the pooled estimator and its outputs are untouched."},
    {"date": "2026-09-28", "post_results": True,
     "change": "R18: attainment-indicator arm (decomposition.indicator_columns; "
               "scripts/18_dummies.py, results/grid_dummies, results/dummies_summary.json): "
               "pooled-reference decomposition with the attainment code entered as one "
               "indicator per observed value (lowest dropped) plus age, premarket only, "
               "full sample, uncorrected because the errors-in-variables correction is "
               "not defined for a set of indicators",
     "reason": "pre-submission review of 28 September 2026: the schooling code entered as "
               "one linear column. Added post-results; the linear-coding outputs in "
               "results/grid are untouched."},
]


def core_specs():
    for od in OUTCOME_DEFS:
        for cs in COVARIATE_SETS:
            for lr in LEARNERS:
                yield {"outcome_def": od, "covariate_set": cs, "learner": lr}
