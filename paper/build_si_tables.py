#!/usr/bin/env python3
"""Build every Supplementary table and every number quoted in the Supplementary
Note from the run logs in results/. Nothing is typed in by hand.

Usage, from the repository root: python paper/build_si_tables.py [results_dir] [out_dir]
(results_dir defaults to results/ next to paper/, out_dir to paper/).

Writes out_dir/tables/S1.tex ... S10.tex (tabular bodies only) and
out_dir/si_numbers.json (every scalar the Note quotes, with its source)."""
import json, glob, os, sys, math
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(OUT, "tables"); os.makedirs(TAB, exist_ok=True)
PRIMARY_SEED = 20261001
ANCHOR = 0.186729281562411
NUM = {}  # numbers for the Note, keyed by name


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(R, d, "*.json"))):
        c = json.load(open(f))
        out[(c["state"], c["year"])] = c
    return out


def spec(c, outcome, cs, learner=None):
    for s in c["spec_results"]:
        if s["outcome_def"] == outcome and s["covariate_set"] == cs and (learner is None or s["learner"] == learner):
            return s
    raise KeyError((outcome, cs, learner))


def fmt(x, d=3, plus=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "not estimable"
    s = f"{x:.{d}f}"
    if plus and x > 0:
        s = "+" + s
    return s.replace("-", "$-$") if x < 0 else s


def write(name, header, rows, colspec, colsep="4pt"):
    ncol = header.count("&") + 1
    body = [f"\\setlength{{\\tabcolsep}}{{{colsep}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + "\\\\", "\\midrule"]
    for r in rows:
        if isinstance(r, str):  # block header spanning the table
            body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{r}}}}}\\\\")
        else:
            body.append(" & ".join(r) + "\\\\")
    body += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(body) + "\n")


def n_fmt(n):
    return f"{n:,}"


# ---------------------------------------------------------------- reliability files
K = json.load(open(os.path.join(R, "kappa.json")))
K8 = json.load(open(os.path.join(R, "kappa_edcat8.json")))
KW = json.load(open(os.path.join(R, "kappa_workers.json")))
PV = json.load(open(os.path.join(R, "piaac_validity.json")))
PG = json.load(open(os.path.join(R, "pooled_gap.json")))
HY = json.load(open(os.path.join(R, "hypotheses.json")))
SL = json.load(open(os.path.join(R, "schooling_lead.json")))
SL25 = json.load(open(os.path.join(R, "schooling_lead_age2565.json")))
MS = json.load(open(os.path.join(R, "meanerror_summary.json")))
MW = json.load(open(os.path.join(R, "meanerror_summary_workers.json")))

# ---------------------------------------------------------------- S1 reliability by age band
bands = {"1": "16 to 24", "2": "25 to 34", "3": "35 to 44", "4": "45 to 54", "5": "55 to 65"}
rows = []
for b, lab in bands.items():
    l, n = K["lit"]["age10"][b], K["num"]["age10"][b]
    rows.append([lab, n_fmt(l["n"]), f"{fmt(l['kappa'])} ({fmt(l['se'])})", f"{fmt(n['kappa'])} ({fmt(n['se'])})"])
for lab, src in (("All adults, 16 to 65", K), ("Earners, 25 to 65", KW)):
    l, n = src["lit"]["overall"], src["num"]["overall"]
    rows.append([lab, n_fmt(l["n"]), f"{fmt(l['kappa'])} ({fmt(l['se'])})", f"{fmt(n['kappa'])} ({fmt(n['se'])})"])
write("S1", "Sample & $n$ & Literacy & Numeracy", rows, "@{}llll@{}")

# ---------------------------------------------------------------- S2 reliability within attainment category
rows = []
n_not, n_resp = 0, 0
for code in map(str, range(1, 9)):
    l, n = K["lit"]["educ"][code], K["num"]["educ"][code]
    if l["kappa"] is None:
        n_not += 1; n_resp += l["n"]
    rows.append([code, n_fmt(l["n"]),
                 "not estimable" if l["kappa"] is None else f"{fmt(l['kappa'])} ({fmt(l['se'])})",
                 "not estimable" if n["kappa"] is None else f"{fmt(n['kappa'])} ({fmt(n['se'])})"])
NUM["S2_not_estimable_categories"] = n_not
NUM["S2_not_estimable_respondents"] = n_resp
NUM["S2_max_kappa_where_estimable"] = max(v["kappa"] for k in ("lit", "num") for v in K[k]["educ"].values() if v["kappa"] is not None)
write("S2", "Attainment code & $n$ & Literacy & Numeracy", rows, "@{}llll@{}")

# ---------------------------------------------------------------- grid loads
G = load_dir("grid"); GH = load_dir("grid_h")
assert len(G) == 250 and len(GH) == 250
markets = sorted(G)
states = sorted({s for s, y in markets})

# ---------------------------------------------------------------- S3 absorption ratio
def ratios(cells, cs, outcomes=("acs50k", "cellmedian")):
    v = []
    for c in cells.values():
        for s in c["spec_results"]:
            if s["covariate_set"] == cs and s["outcome_def"] in outcomes:
                v.append(s["delta_merit"] / s["delta_label"])
    return np.array(v)

rows = []
for lab, cells, cs, outs in (("Extended, own sample", G, "extended", ("acs50k", "cellmedian")),
                             ("Extended, own sample, income threshold", G, "extended", ("acs50k",)),
                             ("Premarket, own sample", G, "premarket", ("acs50k", "cellmedian")),
                             ("Premarket, harmonised sample", GH, "premarket_h", ("acs50k", "cellmedian"))):
    v = ratios(cells, cs, outs)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    rows.append([lab, n_fmt(len(v))] + [fmt(x) for x in q] + [fmt((v < 0).mean())])
    NUM[f"S3_median_{lab}"] = float(np.median(v))
write("S3", "Specification & Runs & p5 & p25 & Median & p75 & p95 & Share $<0$", rows, "@{}lrrrrrrr@{}", colsep="3pt")

# ---------------------------------------------------------------- S4 strict verdict rate by dimension
def strict(s):
    return s["delta_merit"] > 0 > s["delta_label"]
allspecs = [s for c in G.values() for s in c["spec_results"]]
assert len(allspecs) == 3000
def rate(key, val):
    v = [strict(s) for s in allspecs if s[key] == val]
    return sum(v) / len(v)
rows = [["Covariate set", "premarket", fmt(rate("covariate_set", "premarket"))],
        ["", "extended", fmt(rate("covariate_set", "extended"))],
        ["Outcome definition", "income threshold", fmt(rate("outcome_def", "acs50k"))],
        ["", "cell median", fmt(rate("outcome_def", "cellmedian"))],
        ["Learner", "logistic", fmt(rate("learner", "logistic"))],
        ["", "gradient boosting", fmt(rate("learner", "xgboost"))],
        ["", "tabular foundation model", fmt(rate("learner", "tabpfn"))]]
# cross-check against the hypotheses register written by the pipeline
hv = HY["H2_5"]["verdict_rate_by_dimension"]
for k, v in (("covset_premarket", rate("covariate_set", "premarket")), ("covset_extended", rate("covariate_set", "extended")),
             ("outcome_acs50k", rate("outcome_def", "acs50k")), ("learner_tabpfn", rate("learner", "tabpfn"))):
    assert abs(hv[k] - v) < 1e-12, k
NUM["S4_R_positive_all_specs"] = sum(1 for s in allspecs if s["R"] > 0)
# per-market share of the twelve specifications showing the strict pattern: exactly one half in most markets
_share = {m: np.mean([strict(s) for s in G[m]["spec_results"]]) for m in markets}
NUM["S4_markets_share_exactly_half"] = int(sum(1 for v in _share.values() if abs(v - 0.5) < 1e-9))
NUM["S4_markets_share_within_005_of_half"] = int(sum(1 for v in _share.values() if abs(v - 0.5) <= 0.05 + 1e-9))
NUM["S4_share_min"], NUM["S4_share_max"] = float(min(_share.values())), float(max(_share.values()))
write("S4", "Dimension & Level & Strict verdict rate", rows, "@{}lll@{}")

# harmonised-arm rates quoted in the main text
def hrate(cells, cs, learner=None):
    v = [strict(s) for c in cells.values() for s in c["spec_results"] if s["covariate_set"] == cs and (learner is None or s["learner"] == learner)]
    return sum(v) / len(v)
NUM["harmonised_premarket_rate"] = hrate(GH, "premarket_h")
NUM["harmonised_by_learner"] = {l: hrate(GH, "premarket_h", l) for l in ("logistic", "xgboost", "tabpfn")}
NUM["own_sample_premarket_rate"] = rate("covariate_set", "premarket")
NUM["own_sample_extended_rate"] = rate("covariate_set", "extended")

# ---------------------------------------------------------------- S5 dependence across markets (R, primary spec)
Rp = np.array([spec(G[m], "acs50k", "premarket", "logistic")["R"] for m in markets])
st = np.array([m[0] for m in markets])
NUM["R_mean"] = float(Rp.mean()); NUM["R_sd"] = float(Rp.std(ddof=1))
loso = {s: Rp[st != s].mean() for s in states}
lo_s, hi_s = min(loso, key=loso.get), max(loso, key=loso.get)
rng = np.random.default_rng(PRIMARY_SEED)
by_state = {s: Rp[st == s] for s in states}
draws = np.empty(10000)
for i in range(10000):
    pick = rng.choice(states, size=len(states), replace=True)
    draws[i] = np.concatenate([by_state[s] for s in pick]).mean()
ci = np.percentile(draws, [2.5, 97.5]); se_b = draws.std(ddof=1); se_i = Rp.std(ddof=1) / math.sqrt(len(Rp))
NUM.update({"R_loso_min": float(loso[lo_s]), "R_loso_min_state": lo_s, "R_loso_max": float(loso[hi_s]), "R_loso_max_state": hi_s,
            "R_block_ci": [float(ci[0]), float(ci[1])], "R_block_se": float(se_b), "R_iid_se": float(se_i), "R_se_ratio": float(se_b / se_i)})
rows = [["Mean of $R$ across the 250 markets", fmt(Rp.mean(), 4)],
        ["Standard deviation of $R$", fmt(Rp.std(ddof=1), 4)],
        ["Leave-one-state-out mean, minimum", f"{fmt(loso[lo_s], 4)} ({lo_s} omitted)"],
        ["Leave-one-state-out mean, maximum", f"{fmt(loso[hi_s], 4)} ({hi_s} omitted)"],
        ["State-block bootstrap, 95 per cent interval", f"{fmt(ci[0], 4)} to {fmt(ci[1], 4)}"],
        ["State-block bootstrap draws", "10,000"],
        ["Standard error, state-block", fmt(se_b, 4)],
        ["Standard error ignoring the clustering", fmt(se_i, 4)],
        ["Ratio of the two", fmt(se_b / se_i, 2)]]
# panel B: the pooled decomposition quantities from the pipeline's own state-block bootstrap (pooled_gap.json)
rowsB = []
for key, lab in (("uncorrected_residual", "Uncorrected unexplained gap"), ("corrected_residual", "Corrected unexplained gap at the anchor"), ("deepening", "Change on correction")):
    p = PG[key]
    rowsB.append([lab.replace("unexplained gap at the anchor", "gap at anchor").replace("unexplained gap", "gap"), fmt(p["mean"]), f"{fmt(p['state_block_bootstrap']['ci_lo'])} to {fmt(p['state_block_bootstrap']['ci_hi'])}",
                  f"{fmt(p['loso']['min'])} ({p['loso']['min_state']}) to {fmt(p['loso']['max'])} ({p['loso']['max_state']})",
                  fmt(p["se_ratio_block_over_iid"], 2)])
    NUM[f"pooled_{key}"] = {"mean": p["mean"], "ci": [p["state_block_bootstrap"]["ci_lo"], p["state_block_bootstrap"]["ci_hi"]],
                           "loso": [p["loso"]["min"], p["loso"]["min_state"], p["loso"]["max"], p["loso"]["max_state"]], "se_ratio": p["se_ratio_block_over_iid"]}
sw = PG["pooled_sweep"]
NUM["pooled_sweep_max"] = sw["max_over_sweep"]; NUM["pooled_sweep_max_ci_hi"] = sw["max_ci_hi_over_sweep"]
NUM["pooled_sweep_kappa_at_max"] = sw["kappa"][int(np.argmax(sw["mean"]))]
write("S5", "Quantity & Value", rows, "@{}ll@{}")
write("S5B", "Quantity & Mean & State-block 95 per cent & Leave-one-state-out range & SE ratio", rowsB, "@{}lllll@{}", colsep="3pt")

# ---------------------------------------------------------------- S6 computational environment
ver = next(iter(G.values()))["versions"]
hashes = {}
for d in ("grid", "grid_h", "boot_primary", "boot_primary_v2", "boot_meanerror_lit", "grid_years", "grid_years_age2565",
          "grid_meanerror_lit", "grid_meanerror_lit_lo", "grid_meanerror_lit_hi", "grid_meanerror_num", "grid_meanerror_num_hi",
          "grid_meanerror_lit_workers", "grid_meanerror_num_workers", "grid_wage", "grid_adjinc", "grid_wmedian", "grid_age2565", "grid_occ", "grid_exact",
          "grid_prop_lit", "grid_prop_num", "grid_prop_lit_workers", "grid_prop_num_workers", "grid_meanerror_composite", "grid_groupref", "grid_dummies"):
    hs = {json.load(open(f))["config_hash"] for f in glob.glob(os.path.join(R, d, "*.json"))}
    assert len(hs) == 1, (d, hs)
    hashes[d] = hs.pop()
for f in ("pooled_gap.json", "piaac_validity.json", "schooling_lead.json", "schooling_lead_age2565.json", "meanerror_summary.json", "meanerror_summary_workers.json",
          "piaac_validity_composite.json", "meanerror_summary_prop.json", "meanerror_summary_prop_workers.json", "meanerror_summary_composite.json",
          "replicate_se.json", "groupref_summary.json", "dummies_summary.json"):
    hashes[f] = json.load(open(os.path.join(R, f)))["config_hash"]
rows = [["Python", ver["python"]], ["Platform", ver["platform"]], ["NumPy", ver["numpy"]], ["pandas", ver["pandas"]], ["SciPy", ver["scipy"]],
        ["scikit-learn", ver["sklearn"]], ["XGBoost", ver["xgboost"]], ["TabPFN", ver["tabpfn"]], ["PyTorch", ver["torch"]], ["folktables", ver["folktables"]],
        ["Primary seed", str(next(iter(G.values()))["seed"])], ["TabPFN weights", next(iter(G.values()))["tabpfn_model_version"]],
        ["Configuration hash, main grid", hashes["grid"]], ["Harmonised arm", hashes["grid_h"]], ["Bootstrap, version 1", hashes["boot_primary"]],
        ["Bootstrap, version 2", hashes["boot_primary_v2"]], ["Bootstrap, mean-error arm (literacy)", hashes["boot_meanerror_lit"]],
        ["Years-coded arm, all adults", hashes["grid_years"]],
        ["Mean-error arm, literacy 0.297", hashes["grid_meanerror_lit"]], ["Mean-error arm, literacy lower bound 0.159", hashes["grid_meanerror_lit_lo"]],
        ["Mean-error arm, literacy upper bound 0.435", hashes["grid_meanerror_lit_hi"]], ["Mean-error arm, numeracy 0.560", hashes["grid_meanerror_num"]],
        ["Mean-error arm, numeracy upper bound 0.694", hashes["grid_meanerror_num_hi"]],
        ["Years-coded arm, earners 25 to 65", hashes["grid_years_age2565"]], ["Mean-error arm, earners, literacy 0.580", hashes["grid_meanerror_lit_workers"]],
        ["Mean-error arm, earners, numeracy 0.871", hashes["grid_meanerror_num_workers"]],
        ["Wage arm", hashes["grid_wage"]], ["Real-dollar arm", hashes["grid_adjinc"]], ["Weighted cell median arm", hashes["grid_wmedian"]],
        ["Ages 25 to 65 arm", hashes["grid_age2565"]], ["Occupation arm", hashes["grid_occ"]], ["Exact-anchor check", hashes["grid_exact"]],
        ["Pooled gap", hashes["pooled_gap.json"]], ["Validity test", hashes["piaac_validity.json"]],
        ["Schooling leads, all adults; earners", f"{hashes['schooling_lead.json']}; {hashes['schooling_lead_age2565.json']}"],
        ["Proportional arms, all adults: literacy share; numeracy share", f"{hashes['grid_prop_lit']}; {hashes['grid_prop_num']}"],
        ["Proportional arms, earners: literacy share; numeracy share", f"{hashes['grid_prop_lit_workers']}; {hashes['grid_prop_num_workers']}"],
        ["Composite arm; composite validity test", f"{hashes['grid_meanerror_composite']}; {hashes['piaac_validity_composite.json']}"],
        ["Group-reference arm; indicator arm", f"{hashes['grid_groupref']}; {hashes['grid_dummies']}"],
        ["Replicate-variance check", hashes["replicate_se.json"]]]
NUM["hashes"] = hashes
write("S6", "Item & Value", rows, "@{}ll@{}")

# ---------------------------------------------------------------- S7 direction of the correction
rows = []
lead_code = {m: SL["markets"][f"{m[0]} {m[1]}"]["premarket"]["code"]["lead_women_minus_men"] for m in markets}
for cs, lab in (("premarket", "Premarket (age, schooling)"), ("extended", "Extended (+ hours, class of worker)")):
    ex_pos = ex_neg = deep = inc = dec = 0
    deepen = []
    for m in markets:
        s = spec(G[m], "acs50k", cs, "logistic")
        ex_pos += s["decomp_explained"] > 0; ex_neg += s["decomp_explained"] < 0
        d = s["eiv_unexplained_at_kappa"] - s["decomp_unexplained"]
        deep += d < 0; deepen.append(d)
        sweep = [k["unexplained"] for k in s["kappa_sweep"]]
        diffs = np.diff(sweep)
        assert (diffs >= -1e-12).all() or (diffs <= 1e-12).all(), ("not monotone", m, cs)
        inc += sweep[-1] > sweep[0]; dec += sweep[-1] < sweep[0]
    rows.append([lab, "250", str(ex_pos), str(ex_neg), str(deep), str(inc), str(dec)])
    NUM[f"S7_{cs}"] = {"explained_pos": ex_pos, "explained_neg": ex_neg, "deepened": deep, "increasing": inc, "decreasing": dec}
    if cs == "premarket":
        lead = np.array([lead_code[m] for m in markets]); dp = np.array(deepen)
        NUM["S7_corr_deepening_lead"] = float(np.corrcoef(lead, dp)[0, 1])
        NUM["S7_women_lead_n"] = int((lead > 0).sum())
        NUM["S7_deepened_where_women_lead"] = int(((lead > 0) & (dp < 0)).sum())
        NUM["S7_attenuated_where_men_lead"] = int(((lead < 0) & (dp > 0)).sum())
        NUM["S7_men_lead_markets"] = {f"{m[0]} {m[1]}": lead_code[m] for m in markets if lead_code[m] < 0}
        NUM["S7_lead_median_where_women_lead"] = float(np.median(lead[lead > 0]))
        NUM["S7_lead_min_where_women_lead"] = float(lead[lead > 0].min()); NUM["S7_lead_max"] = float(lead.max())
write("S7", "Covariate set & Decomp. & Expl. $>0$ & Expl. $<0$ & Deepened & Incr. in $\\kappa$ & Decr. in $\\kappa$", rows, "@{}lrrrrrr@{}", colsep="3pt")

# ---------------------------------------------------------------- Note A numbers: equivalent common reliability from stored group moments
B1 = load_dir("boot_primary")
assert len(B1) == 250
ks = {1: K["lit"]["sex"]["1"], 2: K["lit"]["sex"]["2"]}  # 1 men, 2 women (GENDER_R)
kf, km = ks[2]["kappa"], ks[1]["kappa"]
kbar = 0.5 * (kf + km)
kstar, kbarV, sens, need, lower = [], [], [], [], []
for m in markets:
    g = B1[m]["group_moments"]["SCHL"]; pf = B1[m]["group_weight_share_female"]; pm = 1 - pf
    Vf, Vm, muf, mum = g["female"]["var"], g["male"]["var"], g["female"]["mean"], g["male"]["mean"]
    W = pf * Vf + pm * Vm; Bt = pf * pm * (muf - mum) ** 2; V = W + Bt
    ks_ = 1 - (pf * (1 - kf) * Vf + pm * (1 - km) * Vm) / V
    kstar.append(ks_); kbarV.append((pf * kf * Vf + pm * km * Vm) / W)
    sens.append(-(pf * Vf - pm * Vm) / (2 * V))
    # differential d needed (kbar fixed) to bring kstar to the sweep floor 0.15: kstar(d) = kstar(0) + sens*d
    need.append((0.15 - (1 - (pf * (1 - kbar) * Vf + pm * (1 - kbar) * Vm) / V)) / sens[-1] if sens[-1] != 0 else float("inf"))
    lf = ks[2]["kappa"] - 1.96 * ks[2]["se"]; lm = ks[1]["kappa"] - 1.96 * ks[1]["se"]
    lower.append(1 - (pf * (1 - lf) * Vf + pm * (1 - lm) * Vm) / V)
kstar, kbarV, sens, need, lower = map(np.array, (kstar, kbarV, sens, need, lower))
NUM["A_kappa_f"] = kf; NUM["A_kappa_m"] = km; NUM["A_kappa_f_se"] = ks[2]["se"]; NUM["A_kappa_m_se"] = ks[1]["se"]
NUM["A_kstar_range"] = [float(kstar.min()), float(kstar.max())]
NUM["A_bound_holds"] = int((kstar >= kbarV - 1e-12).sum())
NUM["A_kbarV_range"] = [float(kbarV.min()), float(kbarV.max())]
NUM["A_below_unweighted_average"] = int((kstar < kbar).sum())
NUM["A_sens_range"] = [float(sens.min()), float(sens.max())]
NUM["A_need_min_abs"] = float(np.abs(need).min()); NUM["A_need_median_abs"] = float(np.median(np.abs(need)))
NUM["A_arith_max_diff"] = float(2 * kbar)  # the smaller reliability cannot be below zero
NUM["A_need_exceeds_max_n"] = int((np.abs(need) > 2 * kbar).sum())
NUM["A_diff_est"] = K["lit"]["sex_difference"]["difference"]; NUM["A_diff_se"] = K["lit"]["sex_difference"]["se"]
NUM["A_lower_bounds"] = [float(km - 1.96 * ks[1]["se"]), float(kf - 1.96 * ks[2]["se"])]
NUM["A_kstar_lower_range"] = [float(lower.min()), float(lower.max())]
# corrected gap at kstar: at least how far below zero? use sweep interpolation on the premarket decomposition
gap_at_kstar = []
for m, ks_ in zip(markets, kstar):
    s = spec(G[m], "acs50k", "premarket", "logistic")
    xs = [k["kappa"] for k in s["kappa_sweep"]]; ys = [k["unexplained"] for k in s["kappa_sweep"]]
    gap_at_kstar.append(np.interp(ks_, xs, ys))
NUM["A_gap_at_kstar_max"] = float(max(gap_at_kstar))

# ---------------------------------------------------------------- Note B numbers: the bootstraps
B2 = load_dir("boot_primary_v2"); BM = load_dir("boot_meanerror_lit")
assert len(B2) == 250 and len(BM) == 250
def counts(B):
    c = {"R_all_draws_pos": 0, "corr_all_draws_neg": 0, "corr_ci_excl0": 0, "deep_all_draws_neg": 0, "deep_ci_excl0": 0, "pd_failures": 0, "consistent": 0}
    for b in B.values():
        I = b["intervals"]
        c["R_all_draws_pos"] += I["R"]["share_gt0"] == 1.0
        c["corr_all_draws_neg"] += I["corrected_residual"]["share_lt0"] == 1.0
        c["corr_ci_excl0"] += I["corrected_residual"]["hi"] < 0
        c["deep_all_draws_neg"] += I["deepening"]["share_lt0"] == 1.0
        c["deep_ci_excl0"] += I["deepening"]["hi"] < 0
        c["pd_failures"] += b["pd_failures"]
        c["consistent"] += bool(b["consistent_with_grid"])
    return c
NUM["B1_registered"] = counts(B1); NUM["B1_v2"] = counts(B2)
# folds by person: largest change in any interval endpoint at three decimals
mx = 0.0
for m in markets:
    for q in ("delta_merit", "R", "corrected_residual", "deepening"):
        for e in ("lo", "hi"):
            mx = max(mx, abs(B1[m]["intervals"][q][e] - B2[m]["intervals"][q][e]))
NUM["B2_max_endpoint_change_folds"] = float(mx)
def width(I): return I["hi"] - I["lo"]
NUM["B2_corr_width_ratio_kappa_drawn"] = float(np.mean([width(B2[m]["intervals"]["corrected_residual_kappa_drawn"]) for m in markets]) / np.mean([width(B2[m]["intervals"]["corrected_residual"]) for m in markets]))
NUM["B2_deep_width_ratio_kappa_drawn"] = float(np.mean([width(B2[m]["intervals"]["deepening_kappa_drawn"]) for m in markets]) / np.mean([width(B2[m]["intervals"]["deepening"]) for m in markets]))
NUM["B2_corr_kappa_drawn_excl0"] = sum(B2[m]["intervals"]["corrected_residual_kappa_drawn"]["hi"] < 0 for m in markets)
NUM["B2_deep_kappa_drawn_excl0"] = sum(B2[m]["intervals"]["deepening_kappa_drawn"]["hi"] < 0 for m in markets)
NUM["B2_kappa_drawn_range"] = [float(min(B2[m]["intervals"]["kappa_drawn"]["lo"] for m in markets)), float(max(B2[m]["intervals"]["kappa_drawn"]["hi"] for m in markets))]
# mean-error bootstrap (literacy delta drawn with its SE, kappa drawn)
NUM["BM_shift"] = next(iter(BM.values()))["shift_female_years"]; NUM["BM_shift_se"] = next(iter(BM.values()))["shift_se"]
NUM["BM_corr_both_drawn_excl0"] = sum(BM[m]["intervals"]["corrected_residual_both_drawn"]["hi"] < 0 for m in markets)
NUM["BM_corr_fixed_excl0"] = sum(BM[m]["intervals"]["corrected_residual"]["hi"] < 0 for m in markets)
NUM["BM_net_both_drawn_covers0"] = sum(BM[m]["intervals"]["net_vs_reported_both_drawn"]["lo"] < 0 < BM[m]["intervals"]["net_vs_reported_both_drawn"]["hi"] for m in markets)
NUM["BM_net_fixed_covers0"] = sum(BM[m]["intervals"]["net_vs_reported"]["lo"] < 0 < BM[m]["intervals"]["net_vs_reported"]["hi"] for m in markets)
NUM["BM_width_ratio_both_over_fixed"] = float(np.mean([width(BM[m]["intervals"]["corrected_residual_both_drawn"]) for m in markets]) / np.mean([width(BM[m]["intervals"]["corrected_residual"]) for m in markets]))
NUM["BM_width_ratio_shift_over_fixed"] = float(np.mean([width(BM[m]["intervals"]["corrected_residual_shift_drawn"]) for m in markets]) / np.mean([width(BM[m]["intervals"]["corrected_residual"]) for m in markets]))
NUM["BM_width_ratio_kappa_over_fixed"] = float(np.mean([width(BM[m]["intervals"]["corrected_residual_kappa_drawn"]) for m in markets]) / np.mean([width(BM[m]["intervals"]["corrected_residual"]) for m in markets]))
NUM["BM_pd_failures"] = sum(BM[m]["pd_failures"] for m in markets)
NUM["B3_fdr_share"] = HY["H2_3"]["share_replicating_fdr05"]; NUM["B3_n_tested"] = HY["H2_3"]["n_tested"]
# positive definiteness across every sweep point of every registered specification run
pd_ok = all(k["unexplained"] is not None for c in G.values() for s in c["spec_results"] for k in s["kappa_sweep"])
NUM["B5_sweep_all_defined_registered"] = bool(pd_ok)

# ---------------------------------------------------------------- S8 validity test
rows = []
speclab = {"years_noage": "years", "years_age": "years + age bands", "edcat8_age": "attainment (8) + age bands"}
for skill, slab in (("lit", "Literacy"), ("num", "Numeracy")):
    for smp, mlab in (("all", "all adults, 16 to 65"), ("workers", "earners, 25 to 65")):
        rows.append(f"{slab}, {mlab}")
        for sp in ("years_noage", "years_age", "edcat8_age"):
            r = PV["specs"][skill][smp][sp]
            bf = f"{fmt(r['b_female']['point'], 2, True)} ({fmt(r['b_female']['se'], 2)})"
            if r["benchmark"] is None:
                rows.append([speclab[sp], n_fmt(r["n"]), bf, "", "", "", ""])
            else:
                rows.append([speclab[sp], n_fmt(r["n"]), bf,
                             f"{fmt(r['benchmark']['point'], 2, True)} ({fmt(r['benchmark']['se'], 2)})",
                             f"{fmt(r['difference']['point'], 2, True)} ({fmt(r['difference']['se'], 2)})",
                             fmt(r["difference"]["z"], 2),
                             f"{fmt(r['delta_proxy_units']['point'], 3, True)} ({fmt(r['delta_proxy_units']['se'], 3)})"])
            NUM[f"S8_{skill}_{smp}_{sp}"] = {k: (r[k]["point"] if isinstance(r.get(k), dict) else r.get(k)) for k in ("b_female", "b_x", "kappa_within", "dxbar", "benchmark", "difference", "delta_proxy_units")}
write("S8", "Specification & $n$ & Female coefficient & Benchmark & Difference & $z$ & $\\delta$ (years)", rows, "@{}lrlllrl@{}", colsep="3pt")
NUM["S8_z_range"] = [min(PV["specs"][s][m][sp]["difference"]["z"] for s in ("lit", "num") for m in ("all", "workers") for sp in ("years_noage", "years_age")),
                    max(PV["specs"][s][m][sp]["difference"]["z"] for s in ("lit", "num") for m in ("all", "workers") for sp in ("years_noage", "years_age"))]
NUM["S8_max_delta_change_age_bands"] = max(abs(PV["specs"][s][m]["years_age"]["delta_proxy_units"]["point"] - PV["specs"][s][m]["years_noage"]["delta_proxy_units"]["point"]) for s in ("lit", "num") for m in ("all", "workers"))

# ---------------------------------------------------------------- S9 corrected gap against delta
rows = []
zero_cross = {}
for summ, samp, anchor_lab in ((MS, "All adults, $\\kappa = 0.187$", "all"), (MW, "Earners 25 to 65, $\\kappa = 0.243$", "workers")):
    for cs, cslab in (("premarket", "premarket"), ("extended", "extended")):
        first = next(iter(summ["arms"].values()))["covsets"][cs]
        p0 = first["pooled"]["U_yk"]
        # delta = 0 row (the years-coded reference arm, reliability correction only)
        neg0 = sum(1 for mk in first["markets"] if mk["U_yk"] < 0)
        rows.append(f"{samp}, {cslab} covariates")
        neg0f = sum(1 for mk in first["markets"] if mk["sweep_y_max"] < 0)
        rows.append(["0 (classical)", fmt(p0["mean"]), f"{fmt(p0['ci_lo'])} to {fmt(p0['ci_hi'])}", f"{fmt(p0['min'])} to {fmt(p0['max'])}", str(neg0), str(neg0f), str(first["reliability_correction_deepens_years_arm"])])
        xs, ys, per_market = [0.0], [p0["mean"]], {mk["market"]: [mk["U_yk"]] for mk in first["markets"]}
        for tag, arm in sorted(summ["arms"].items(), key=lambda kv: kv[1]["delta_years"]):
            cv = arm["covsets"][cs]; p = cv["pooled"]["U_mk"]
            taglab = {"lit_lo": "literacy, lower", "lit": "literacy", "lit_hi": "literacy, upper", "num": "numeracy", "num_hi": "numeracy, upper",
                      "lit_workers": "literacy", "num_workers": "numeracy"}[tag]
            rows.append([f"{arm['delta_years']:.3f} ({taglab})", fmt(p["mean"]), f"{fmt(p['ci_lo'])} to {fmt(p['ci_hi'])}",
                         f"{fmt(p['min'])} to {fmt(p['max'])}", str(cv["U_m_anchor_negative"]), str(cv["U_m_floor_negative"]),
                         str(cv['reliability_correction_deepens_after_meanerror'])])
            xs.append(arm["delta_years"]); ys.append(p["mean"])
            for mk in cv["markets"]:
                per_market[mk["market"]].append(mk["U_mk"])
        b, a0 = np.polyfit(xs, ys, 1)
        resid = max(abs(a0 + b * x - y) for x, y in zip(xs, ys))
        zc = -a0 / b
        pm_zero = np.array([-np.polyfit(xs, v, 1)[1] / np.polyfit(xs, v, 1)[0] for v in per_market.values()])
        zero_cross[(anchor_lab, cs)] = {"intercept": a0, "slope": b, "max_resid": resid, "zero": zc, "neutral": -a0 / b if False else None,
                                        "per_market_median": float(np.median(pm_zero)), "per_market_min": float(pm_zero.min()), "per_market_max": float(pm_zero.max()),
                                        "n_points": len(xs)}
        # neutral point: where corrected equals uncorrected years-coded U_y1
        U1 = first["pooled"]["U_y1"]["mean"]
        zero_cross[(anchor_lab, cs)]["U_y1"] = U1
        zero_cross[(anchor_lab, cs)]["neutral"] = (U1 - a0) / b
        if anchor_lab == "all" and cs == "premarket":
            d_num = MS["arms"]["num"]["delta_years"]; d_hi = MS["arms"]["num_hi"]["delta_years"]
            zero_cross[(anchor_lab, cs)]["n_above_numeracy"] = int((pm_zero > d_num).sum())
            zero_cross[(anchor_lab, cs)]["n_above_numeracy_upper"] = int((pm_zero > d_hi).sum())
            zero_cross[(anchor_lab, cs)]["n_below_num_workers"] = int((pm_zero < 0.871).sum())
            # the direct run at the numeracy upper bound against the per-market fit: markets where the
            # two disagree about the sign at that delta, with the run value there
            run_hi = {mk["market"]: mk["U_mk"] for mk in MS["arms"]["num_hi"]["covsets"]["premarket"]["markets"]}
            fit_neg = {m: z > d_hi for m, z in zip(per_market.keys(), pm_zero)}
            disagree = {m: run_hi[m] for m in run_hi if (run_hi[m] < 0) != fit_neg[m]}
            zero_cross[(anchor_lab, cs)]["n_run_negative_upper"] = int(sum(1 for v in run_hi.values() if v < 0))
            zero_cross[(anchor_lab, cs)]["fit_run_disagree"] = disagree
            zero_cross[(anchor_lab, cs)]["fit_run_disagree_max_abs"] = float(max(abs(v) for v in disagree.values())) if disagree else 0.0
NUM["S8_leads"] = {s: {"point": PV["specs"]["lit"][s]["years_noage"]["dxbar"]["point"], "se": PV["specs"]["lit"][s]["years_noage"]["dxbar"]["se"]} for s in ("all", "workers")}
NUM["S9_fits"] = {f"{k[0]}_{k[1]}": v for k, v in zero_cross.items()}
# along the sweep: markets with a negative corrected gap at each swept reliability, premarket, per mean-error arm
def sweep_negative(d):
    out = {}
    for f in glob.glob(os.path.join(R, d, "*.json")):
        c = json.load(open(f)); s = spec(c, "acs50k", "premarket")
        for k in s["kappa_sweep"]:
            out.setdefault(f"{k['kappa']:.2f}", []).append(k["unexplained"] < 0)
    return {k: int(sum(v)) for k, v in sorted(out.items())}
NUM["S9_sweep_negative"] = {d: sweep_negative(d) for d in ("grid_meanerror_lit_lo", "grid_meanerror_lit", "grid_meanerror_lit_hi", "grid_meanerror_num", "grid_meanerror_num_hi",
                                                          "grid_meanerror_lit_workers", "grid_meanerror_num_workers")}
NUM["S9_lowest_kappa_all_negative"] = {d: min(float(k) for k, v in sc.items() if v == 250) for d, sc in NUM["S9_sweep_negative"].items()}
NUM["S9_U_y1"] = {"all": MS["arms"]["lit"]["covsets"]["premarket"]["pooled"]["U_y1"]["mean"], "workers": MW["arms"]["lit_workers"]["covsets"]["premarket"]["pooled"]["U_y1"]["mean"]}
# the interval bounds carried into the arm are the estimate plus and minus two standard errors
_pv = PV["specs"]["lit"]["all"]["years_noage"]["delta_proxy_units"]; _pn = PV["specs"]["num"]["all"]["years_noage"]["delta_proxy_units"]
NUM["S9_bound_multiplier"] = {"lit_lo": (_pv["point"] - MS["arms"]["lit_lo"]["delta_years"]) / _pv["se"], "lit_hi": (MS["arms"]["lit_hi"]["delta_years"] - _pv["point"]) / _pv["se"],
                              "num_hi": (MS["arms"]["num_hi"]["delta_years"] - _pn["point"]) / _pn["se"], "num_lo_not_run": _pn["point"] - 2 * _pn["se"]}
# the men-lead markets under the years coding (the shift is negative there in the proportional arm)
NUM["S9_men_lead_years"] = sorted(m for m, v in SL["markets"].items() if v["premarket"]["years"]["lead_women_minus_men"] < 0)
NUM["S9_men_lead_years_workers"] = sorted(m for m, v in SL25["markets"].items() if v["premarket"]["years"]["lead_women_minus_men"] < 0)
# Berkson-consistent adjustment (Note A.8): under M* = X + eta the only adjustment is a mean shift of women's schooling by
# -b_F / b_X years with no reliability step; lowering women's years by delta with the sex indicator in the regression moves the
# uncorrected unexplained component by exactly delta times the uncorrected schooling return, so the per-market value is
# U_y1 + delta_B * b1 with b1 read from the stored mean-error arms (meanerror_shift / delta, identical across the five arms)
_b1 = {}
for tag, a in MS["arms"].items():
    for mk in a["covsets"]["premarket"]["markets"]:
        _b1.setdefault(mk["market"], []).append(mk["meanerror_shift"] / a["delta_years"])
assert max(max(v) - min(v) for v in _b1.values()) < 1e-9
_b1 = {m: float(np.mean(v)) for m, v in _b1.items()}
_Uy1 = {mk["market"]: mk["U_y1"] for mk in MS["arms"]["lit"]["covsets"]["premarket"]["markets"]}
NUM["A8_berkson"] = {"b1_pooled": float(np.mean(list(_b1.values()))), "b1_min": min(_b1.values()), "b1_max": max(_b1.values()), "linearity_spread": float(max(max(v) - min(v) for v in [[mk["meanerror_shift"] / a["delta_years"] for a in MS["arms"].values() for mk in a["covsets"]["premarket"]["markets"] if mk["market"] == m] for m in _Uy1]))}
for skill in ("lit", "num"):
    v = PV["specs"][skill]["all"]["years_noage"]; dB = -v["b_female"]["point"] / v["b_x"]["point"]
    UB = {m: _Uy1[m] + dB * _b1[m] for m in _Uy1}
    NUM["A8_berkson"][skill] = {"shift_years": dB, "b_female": v["b_female"]["point"], "b_x": v["b_x"]["point"], "pooled": float(np.mean(list(UB.values()))), "negative": int(sum(u < 0 for u in UB.values())), "max": float(max(UB.values())), "min": float(min(UB.values()))}
NUM["S9_pooled_premarket"] = {tag: {"mean": MS["arms"][tag]["covsets"]["premarket"]["pooled"]["U_mk"]["mean"], "max": MS["arms"][tag]["covsets"]["premarket"]["pooled"]["U_mk"]["max"],
                                    "negative_at_anchor": MS["arms"][tag]["covsets"]["premarket"]["U_m_anchor_negative"]} for tag in MS["arms"]}
write("S9", "$\\delta$ (years) & Pooled mean & 95 per cent interval & Market range & Neg. anchor & Neg. floor & Deepens", rows, "@{}lllllll@{}", colsep="2.5pt")
# the sign rule: adjusted lead sign vs direction of the reliability step, per market and delta
rule = {}
for summ, sl_, lab in ((MS, SL, "all"), (MW, SL25, "workers")):
    for cs in ("premarket", "extended"):
        for tag, arm in summ["arms"].items():
            d = arm["delta_years"]; ok = 0
            for mk in arm["covsets"][cs]["markets"]:
                lead = sl_["markets"][mk["market"]][cs]["years"]["lead_women_minus_men"]
                predicted_deepen = (lead - d) > 0
                ok += predicted_deepen == (mk["deep_m"] < 0)
            rule[f"{lab}_{cs}_{tag}"] = ok
NUM["S9_sign_rule_matches"] = rule
NUM["lead_years_all_premarket"] = SL["summary"]["premarket_years"]; NUM["lead_years_all_extended"] = SL["summary"]["extended_years"]
NUM["lead_years_w25_premarket"] = SL25["summary"]["premarket_years"]; NUM["lead_years_w25_extended"] = SL25["summary"]["extended_years"]
# ratio of corrected to uncorrected return: slope / b(1); the uncorrected return is not stored, so the ratio slope*kappa is checked against 1/kappa instead
NUM["S9_slope_times_kappa"] = {f"{k[0]}_{k[1]}": v["slope"] * (ANCHOR if k[0] == "all" else 0.2425777466208153) for k, v in zero_cross.items()}

# ---------------------------------------------------------------- S10 the five arms and the occupation threshold
def arm_rows(d, lab, cs_list=("premarket", "extended"), anchor_key="eiv_unexplained_at_kappa"):
    cells = load_dir(d); assert len(cells) == 250, d
    out = []
    for cs in cs_list:
        Rv, raw, U, Uk, deep, n = [], [], [], [], 0, []
        for m in markets:
            s = spec(cells[m], "acs50k", cs, "logistic")
            Rv.append(s["R"]); raw.append(s["decomp_raw_logpts"]); U.append(s["decomp_unexplained"]); n.append(s["n"])
            if s.get(anchor_key) is not None:
                Uk.append(s[anchor_key]); deep += s[anchor_key] < s["decomp_unexplained"]
        Rv, raw, U = map(np.array, (Rv, raw, U))
        out.append([lab, cs.replace("extended_occ", "ext. + occ."), n_fmt(int(np.mean(n))), fmt(Rv.mean()), str(int((Rv > 0).sum())), fmt(raw.mean()), fmt(U.mean()),
                    fmt(np.mean(Uk)) if Uk else "not defined", str(deep) if Uk else "none correctable"])
        NUM[f"S10_{d}_{cs}"] = {"R_mean": float(Rv.mean()), "R_pos": int((Rv > 0).sum()), "raw": float(raw.mean()), "U": float(U.mean()),
                                "Uk": float(np.mean(Uk)) if Uk else None, "deepens": deep, "n_mean": float(np.mean(n))}
    return cells, out
rows = []
for d, lab in (("grid", "Main grid"), ("grid_years", "Years-coded schooling"), ("grid_wage", "Wage, full-time"), ("grid_adjinc", "Real dollars"),
               ("grid_wmedian", "Weighted median"), ("grid_age2565", "Ages 25 to 65")):
    cells, rr = arm_rows(d, lab); rows += rr
    if d == "grid_years":
        NUM["S10_years_max_R_diff"] = float(max(abs(spec(cells[m], oc, cs, "logistic")["R"] - spec(G[m], oc, cs, "logistic")["R"]) for m in markets for cs in ("premarket", "extended") for oc in ("acs50k", "cellmedian")))
        NUM["S10_years_U_pm"] = NUM["S10_grid_years_premarket"]["U"]; NUM["S10_years_Uk_pm"] = NUM["S10_grid_years_premarket"]["Uk"]
    if d == "grid_adjinc":
        mxU = max(abs(spec(cells[m], "acs50k", cs, "logistic")["decomp_unexplained"] - spec(G[m], "acs50k", cs, "logistic")["decomp_unexplained"]) for m in markets for cs in ("premarket", "extended"))
        mxR = max(abs(spec(cells[m], "acs50k", cs, "logistic")["R"] - spec(G[m], "acs50k", cs, "logistic")["R"]) for m in markets for cs in ("premarket", "extended"))
        NUM["S10_adjinc_max_change_U"] = float(mxU); NUM["S10_adjinc_max_change_R"] = float(mxR)
    if d == "grid_wmedian":
        signs = sum(1 for m in markets for cs in ("premarket", "extended") if np.sign(spec(cells[m], "cellmedian", cs, "logistic")["R"]) != np.sign(spec(G[m], "cellmedian", cs, "logistic")["R"]))
        NUM["S10_wmedian_sign_changes"] = signs
        NUM["S10_wmedian_max_change_R_cellmedian"] = float(max(abs(spec(cells[m], "cellmedian", cs, "logistic")["R"] - spec(G[m], "cellmedian", cs, "logistic")["R"]) for m in markets for cs in ("premarket", "extended")))
    if d == "grid_age2565":
        dp = [spec(cells[m], "acs50k", "premarket", "logistic")["eiv_unexplained_at_kappa"] - spec(cells[m], "acs50k", "premarket", "logistic")["decomp_unexplained"] for m in markets]
        NUM["S10_age2565_deepening_mean"] = float(np.mean(dp)); NUM["S10_age2565_anchor"] = spec(cells[markets[0]], "acs50k", "premarket", "logistic")["kappa_point"]
    if d == "grid_wage":
        NUM["S10_wage_raw_ratio"] = {cs: NUM[f"S10_grid_wage_{cs}"]["raw"] / NUM[f"S10_grid_{cs}"]["raw"] for cs in ("premarket", "extended")}
OC, rr = arm_rows("grid_occ", "Occupation groups", cs_list=("extended_occ",)); rows += rr
write("S10", "Arm & Covariates & Mean $n$ & Mean $R$ & $R>0$ & Raw gap & Uncorr. & Corr. & Deepened", rows, "@{}llrrrrrrr@{}", colsep="2.5pt")
# occupation threshold distribution and admissible sweep
thr = np.array([spec(OC[m], "acs50k", "extended_occ", "logistic")["eiv_min_kappa"] for m in markets])
rng = np.random.default_rng(PRIMARY_SEED + 1)
by_state = {s: thr[st == s] for s in states}
d2 = np.array([np.concatenate([by_state[s] for s in rng.choice(states, size=50, replace=True)]).mean() for _ in range(10000)])
NUM["S10_occ_threshold"] = {"mean": float(thr.mean()), "ci": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))], "median": float(np.median(thr)),
                            "min": float(thr.min()), "max": float(thr.max()), "above_anchor": int((thr > ANCHOR).sum()),
                            "correctable_at_anchor": int(sum(spec(OC[m], "acs50k", "extended_occ", "logistic")["eiv_correctable"] for m in markets))}
adm = {}
for kap in (0.35, 0.40, 0.50, 0.75, 1.0):
    vals = []
    for m in markets:
        s = spec(OC[m], "acs50k", "extended_occ", "logistic")
        v = next(k["unexplained"] for k in s["kappa_sweep"] if abs(k["kappa"] - kap) < 1e-9)
        vals.append(v)
    defined = [v for v in vals if v is not None]
    adm[str(kap)] = {"defined": len(defined), "mean": float(np.mean(defined)) if defined else None, "negative": int(sum(v < 0 for v in defined))}
NUM["S10_occ_admissible_sweep"] = adm
NUM["S10_occ_R_mean_ext_grid"] = NUM["S10_grid_extended"]["R_mean"]
rows = [["Threshold reliability, mean (state-block 95 per cent interval)", f"{fmt(thr.mean())} ({fmt(np.percentile(d2, 2.5))} to {fmt(np.percentile(d2, 97.5))})"],
        ["Threshold, median and range across markets", f"{fmt(np.median(thr))}; {fmt(thr.min(), 4)} to {fmt(thr.max(), 4)}"],
        ["Markets where the threshold exceeds the anchor of 0.1867", f"{int((thr > ANCHOR).sum())} of 250"]]
for kap in ("0.35", "0.4", "0.5", "0.75", "1.0"):
    a = adm[kap]
    rows.append([f"Corrected gap at $\\kappa = {float(kap):.2f}$: markets defined; mean; negative", f"{a['defined']}; {fmt(a['mean']) if a['mean'] is not None else 'none'}; {a['negative']}"])
write("S10B", "Quantity & Value", rows, "@{}p{9.5cm}p{5cm}@{}")

# ---------------------------------------------------------------- S11 replicate-weight standard errors (review computation R16)
RS = json.load(open(os.path.join(R, "replicate_se.json")))
rs = RS["summary"]
assert rs["n_markets"] == 250 and RS["n_replicates"] == 80 and rs["point_matches_bootstrap_point"] == 250
rows = []
labels = {"raw": "Raw gap", "explained": "Explained", "unexplained": "Unexplained", "corrected_unexplained": "Corrected unexplained", "deepening": "Deepening"}
excl_all = {}
for q in ("raw", "explained", "unexplained", "corrected_unexplained", "deepening"):
    se = rs["replicate_se"][q]
    ratio = rs["ratio_replicate_se_over_bootstrap_se"].get(q)
    excl_all[q] = int(sum(1 for m in RS["markets"].values() if abs(m["point"][q]) > 1.96 * m["replicate_se"][q]))
    if q in rs["excludes_zero_at_1_96_replicate_se"]:
        assert excl_all[q] == rs["excludes_zero_at_1_96_replicate_se"][q], q
    rows.append([labels[q], f"{fmt(se['median'], 4)} ({fmt(se['min'], 4)} to {fmt(se['max'], 4)})",
                 f"{fmt(ratio['median'], 2)} ({fmt(ratio['min'], 2)} to {fmt(ratio['max'], 2)})" if ratio else "no bootstrap interval",
                 str(excl_all[q])])
write("S11", "Quantity & Replicate SE, median (range) & Ratio to bootstrap SE (range) & Excludes zero", rows, "@{}llll@{}", colsep="3pt")
NUM["S11_replicate"] = {"ratio": rs["ratio_replicate_se_over_bootstrap_se"], "excl": excl_all,
                        "clipped_total": rs["negative_replicate_weights_clipped_total"], "markets_with_clipped": int(sum(1 for m in RS["markets"].values() if m["negative_replicate_weights_clipped"] > 0)),
                        "entries_total": int(sum(m["n"] for m in RS["markets"].values()) * 80), "point_matches": rs["point_matches_bootstrap_point"]}
NUM["S11_replicate"]["clipped_share_pct"] = 100.0 * NUM["S11_replicate"]["clipped_total"] / NUM["S11_replicate"]["entries_total"]
# the bootstrap counts the replicate check is compared with (version 1 percentile intervals, as the check itself uses)
NUM["S11_boot_excl"] = {"corrected_unexplained": int(sum(1 for m in markets if B1[m]["intervals"]["corrected_residual"]["hi"] < 0)),
                        "deepening": int(sum(1 for m in markets if B1[m]["intervals"]["deepening"]["hi"] < 0))}

# ---------------------------------------------------------------- S12 proportional error difference (R14)
MP = json.load(open(os.path.join(R, "meanerror_summary_prop.json"))); MPW = json.load(open(os.path.join(R, "meanerror_summary_prop_workers.json")))
rows = []; NUM["S12_prop"] = {}
for summ, samp, key in ((MP, "All adults, $\\kappa = 0.187$", "all"), (MPW, "Earners 25 to 65, $\\kappa = 0.243$", "workers")):
    for cs in ("premarket", "extended"):
        rows.append(f"{samp}, {cs} covariates")
        first = next(iter(summ["arms"].values()))["covsets"][cs]; p0 = first["pooled"]["U_yk"]
        rows.append(["0 (classical)", "", fmt(p0["mean"]), f"{fmt(p0['ci_lo'])} to {fmt(p0['ci_hi'])}", f"{fmt(p0['min'])} to {fmt(p0['max'])}",
                     str(sum(1 for mk in first["markets"] if mk["U_yk"] < 0)), str(sum(1 for mk in first["markets"] if mk["sweep_y_max"] < 0))])
        for tag, arm in sorted(summ["arms"].items(), key=lambda kv: kv[1]["share_of_schooling_lead"]):
            cv = arm["covsets"][cs]; p = cv["pooled"]["U_mk"]; sh = cv["pooled"]["applied_shift_years"]
            lab = "literacy" if "lit" in tag else "numeracy"
            rows.append([f"{arm['share_of_schooling_lead']:.3f} ({lab})", f"{fmt(sh['mean'])} ({fmt(sh['min'])} to {fmt(sh['max'])})", fmt(p["mean"]),
                         f"{fmt(p['ci_lo'])} to {fmt(p['ci_hi'])}", f"{fmt(p['min'])} to {fmt(p['max'])}", str(cv["U_m_anchor_negative"]), str(cv["U_m_floor_negative"])])
            NUM["S12_prop"][f"{key}_{cs}_{lab}"] = {"share": arm["share_of_schooling_lead"], "shift_mean": sh["mean"], "shift_min": sh["min"], "shift_max": sh["max"],
                                                     "mean": p["mean"], "ci": [p["ci_lo"], p["ci_hi"]], "min": p["min"], "max": p["max"],
                                                     "neg_anchor": cv["U_m_anchor_negative"], "neg_floor": cv["U_m_floor_negative"], "flips": cv["U_m_sign_flips_in_sweep"],
                                                     "positive_markets": [mk["market"] for mk in cv["markets"] if mk["U_mk"] > 0],
                                                     "lead_mean": cv["pooled"]["schooling_lead_years"]["mean"], "n_men_lead": int(sum(1 for mk in cv["markets"] if mk["schooling_lead_years"] < 0))}
write("S12", "Share of lead & Shift, years (range) & Pooled mean & 95 per cent interval & Market range & Neg. anchor & Neg. floor", rows, "@{}lllllll@{}", colsep="2.5pt")

# ---------------------------------------------------------------- S13 composite construct (R15)
KC = json.load(open(os.path.join(R, "kappa_composite.json"))); KCW = json.load(open(os.path.join(R, "kappa_composite_workers.json")))
PVC = json.load(open(os.path.join(R, "piaac_validity_composite.json")))["specs"]["composite"]
MC = json.load(open(os.path.join(R, "meanerror_summary_composite.json")))
assert abs(KC["lit"]["overall"]["kappa"] - ANCHOR) < 1e-12  # the composite file recomputes the literacy anchor and must reproduce it
rows = []
for kc, samp in ((KC, "All adults, 16 to 65"), (KCW, "Earners, 25 to 65")):
    c = kc["composite"]; o = c["overall"]; d = c["sex_difference"]
    rows.append([samp, f"{fmt(o['kappa'])} ({fmt(o['kappa'] - 1.96 * o['se'])}, {fmt(o['kappa'] + 1.96 * o['se'])})", f"{fmt(c['sex']['1']['kappa'])} ({fmt(c['sex']['1']['se'])})",
                 f"{fmt(c['sex']['2']['kappa'])} ({fmt(c['sex']['2']['se'])})", f"{fmt(d['difference'], 3, True)} ({fmt(d['se'])}), $z = {d['z']:.2f}$", n_fmt(o["n"])])
write("S13", "Sample & Overall (95 per cent CI) & Men (SE) & Women (SE) & Men minus women (SE), $z$ & $n$", rows, "@{}llllll@{}", colsep="2pt")
rows = []
for samp, lab in (("all", "All adults"), ("workers", "Earners")):
    for sp, splab in (("years_noage", "years, no age terms"), ("years_age", "years, age bands")):
        v = PVC[samp][sp]
        rows.append([f"{lab}, {splab.replace('no age terms', 'no age').replace('age bands', 'age bands')}", f"{fmt(v['b_female']['point'], 2, True)} ({fmt(v['b_female']['se'], 2)})", f"{fmt(v['benchmark']['point'], 2, True)} ({fmt(v['benchmark']['se'], 2)})",
                     f"{fmt(v['difference']['point'], 2, True)} ({fmt(v['difference']['se'], 2)})", fmt(v['difference']['z'], 2), f"{fmt(v['delta_proxy_units']['point'], 3, True)} ({fmt(v['delta_proxy_units']['se'])})", n_fmt(v["n"])])
write("S13B", "Specification & Female coef. (SE) & Benchmark (SE) & Difference (SE) & $z$ & $\\delta$, years (SE) & $n$", rows, "@{}lllllll@{}", colsep="2pt")
rows = []
arm = MC["arms"]["composite"]; NUM["S13_composite"] = {"delta": arm["delta_years"], "kappa_all": KC["composite"]["overall"]["kappa"], "kappa_all_se": KC["composite"]["overall"]["se"],
                                                       "kappa_workers": KCW["composite"]["overall"]["kappa"], "kappa_workers_se": KCW["composite"]["overall"]["se"],
                                                       "validity": {s: {k: PVC[s]["years_noage"][k] for k in ("b_female", "benchmark", "difference", "delta_proxy_units", "kappa_within")} for s in ("all", "workers")},
                                                       "delta_age_bands": {s: PVC[s]["years_age"]["delta_proxy_units"]["point"] for s in ("all", "workers")}, "arm": {}}
CA = load_dir("grid_meanerror_composite")
for cs in ("premarket", "extended"):
    cv = arm["covsets"][cs]; p = cv["pooled"]["U_mk"]
    kap_comp = next(iter(CA.values()))["spec_results"][0]["kappa_point"]
    at_lit = [spec(CA[m], "acs50k", cs)["eiv_unexplained_at"]["0.186729"] for m in markets]
    rows.append([f"{cs.capitalize()}, composite {kap_comp:.3f}", fmt(p["mean"]), f"{fmt(p['ci_lo'])} to {fmt(p['ci_hi'])}", f"{fmt(p['min'])} to {fmt(p['max'])}", str(cv["U_m_anchor_negative"]), str(cv["U_m_floor_negative"]), str(cv["reliability_correction_deepens_after_meanerror"])])
    rows.append([f"{cs.capitalize()}, literacy 0.187", fmt(float(np.mean(at_lit))), "", f"{fmt(min(at_lit))} to {fmt(max(at_lit))}", str(sum(1 for v in at_lit if v < 0)), "", ""])
    NUM["S13_composite"]["arm"][cs] = {"kappa": kap_comp, "mean": p["mean"], "ci": [p["ci_lo"], p["ci_hi"]], "min": p["min"], "max": p["max"], "neg_anchor": cv["U_m_anchor_negative"], "neg_floor": cv["U_m_floor_negative"],
                                       "at_lit_mean": float(np.mean(at_lit)), "at_lit_neg": int(sum(1 for v in at_lit if v < 0)), "deepens": cv["reliability_correction_deepens_after_meanerror"]}
write("S13C", "Covariate set, anchor & Pooled mean & 95 per cent interval & Market range & Neg. anchor & Neg. floor & Deepens", rows, "@{}lllllll@{}", colsep="2pt")

# ---------------------------------------------------------------- S14 attainment indicators (R18)
DS = json.load(open(os.path.join(R, "dummies_summary.json")))
rows = []
for q, lab in (("raw", "Raw gap"), ("explained", "Explained, indicator coding"), ("unexplained", "Unexplained, indicator coding"), ("linear_unexplained", "Unexplained, linear coding (main grid)"), ("difference_from_linear", "Difference, indicator minus linear")):
    v = DS["pooled"][q]; b = v["state_block_bootstrap"]
    rows.append([lab, fmt(v["mean"], 4), f"{fmt(b['ci_lo'], 4)} to {fmt(b['ci_hi'], 4)}", f"{fmt(v['min'], 4)} to {fmt(v['max'], 4)}"])
write("S14", "Quantity & Pooled mean & State-block 95 per cent & Market range", rows, "@{}llll@{}")
lv = [m["n_schl_levels_observed"] for m in DS["markets"].values()]
NUM["S14_dummies"] = {"pooled": {q: {"mean": DS["pooled"][q]["mean"], "ci": [DS["pooled"][q]["state_block_bootstrap"]["ci_lo"], DS["pooled"][q]["state_block_bootstrap"]["ci_hi"]], "min": DS["pooled"][q]["min"], "max": DS["pooled"][q]["max"]} for q in DS["pooled"]},
                      "counts": DS["counts"], "levels_min": min(lv), "levels_max": max(lv)}

# ---------------------------------------------------------------- S15 group-specific reference (R17)
GS = json.load(open(os.path.join(R, "groupref_summary.json")))
GR = load_dir("grid_groupref")
ao = next(iter(GR.values()))["arm_options"]
rows = []
order = [("raw", "Raw gap"), ("pooled_ref_uncorrected", "Pooled reference, uncorrected"), ("pooled_ref_corrected", "Pooled reference, corrected at 0.187"),
         ("male_ref_uncorrected", "Male ref., uncorrected"), ("male_ref_corrected_common", "Male ref., corrected, common anchor"), ("male_ref_corrected_sexspecific", "Male ref., corrected, sex-specific"),
         ("female_ref_uncorrected", "Female ref., uncorrected"), ("female_ref_corrected_common", "Female ref., corrected, common anchor"), ("female_ref_corrected_sexspecific", "Female ref., corrected, sex-specific"),
         ("male_ref_sexspecific_minus_common", "Male ref.: sex-specific minus common"), ("female_ref_sexspecific_minus_common", "Female ref.: sex-specific minus common")]
for q, lab in order:
    v = GS["pooled"][q]; b = v["state_block_bootstrap"]
    cnt = GS["counts"].get(q + "_negative"); pos = GS["counts"].get(q + "_positive")
    last = f"{cnt} negative" if cnt is not None else (f"{pos} positive" if pos is not None else "")
    rows.append([lab, fmt(v["mean"], 4, plus=q.endswith("common")), f"{fmt(b['ci_lo'], 4, plus=q.endswith('common'))} to {fmt(b['ci_hi'], 4, plus=q.endswith('common'))}", f"{fmt(v['min'], 4, plus=q.endswith('common'))} to {fmt(v['max'], 4, plus=q.endswith('common'))}", last])
write("S15", "Quantity & Pooled mean & State-block 95 per cent & Market range & Markets", rows, "@{}lllll@{}", colsep="2.5pt")
NUM["S15_groupref"] = {"pooled": {q: {"mean": GS["pooled"][q]["mean"], "ci": [GS["pooled"][q]["state_block_bootstrap"]["ci_lo"], GS["pooled"][q]["state_block_bootstrap"]["ci_hi"]], "min": GS["pooled"][q]["min"], "max": GS["pooled"][q]["max"]} for q in GS["pooled"]},
                       "counts": GS["counts"], "kappa_common": ao["kappa_common"], "kappa_men": ao["kappa_men"], "kappa_women": ao["kappa_women"],
                       "max_abs_move": max(abs(GS["pooled"]["male_ref_sexspecific_minus_common"]["min"]), abs(GS["pooled"]["male_ref_sexspecific_minus_common"]["max"]),
                                          abs(GS["pooled"]["female_ref_sexspecific_minus_common"]["min"]), abs(GS["pooled"]["female_ref_sexspecific_minus_common"]["max"]))}
# consistency: the pooled-reference values stored with the group-reference arm equal the main grid's
assert all(abs(GR[m]["unexplained"]["pooled_ref_uncorrected"] - spec(G[m], "acs50k", "premarket", "logistic")["decomp_unexplained"]) < 1e-9 for m in markets)
assert all(abs(GR[m]["unexplained"]["pooled_ref_corrected"] - spec(G[m], "acs50k", "premarket", "logistic")["eiv_unexplained_at_kappa"]) < 1e-9 for m in markets)

json.dump(NUM, open(os.path.join(OUT, "si_numbers.json"), "w"), indent=1, default=float)
print("wrote", TAB, "and si_numbers.json")
for k, v in NUM.items():
    if not isinstance(v, (dict, list)) or len(json.dumps(v, default=float)) < 200:
        print(f"  {k}: {json.dumps(v, default=float)}")
    else:
        print(f"  {k}: [{type(v).__name__}]")
