#!/usr/bin/env python3
"""Check every quantitative claim in the main text against the run logs.
Each check prints PASS/FAIL with the computed value; nothing is asserted so
that the whole list runs. Run from the repository root:

    python paper/verify_numbers.py [results_dir]

results_dir defaults to results/ next to paper/. The script reads run logs
only and writes nothing."""
import json, glob, os, sys, math
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ANCHOR = 0.186729281562411


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(R, d, "*.json"))):
        c = json.load(open(f)); out[(c["state"], c["year"])] = c
    return out


def spec(c, outcome, cs, learner=None):
    for s in c["spec_results"]:
        if s["outcome_def"] == outcome and s["covariate_set"] == cs and (learner is None or s["learner"] == learner):
            return s
    raise KeyError((outcome, cs, learner))


results = []
def check(label, claim, value, tol=None, kind="eq"):
    if kind == "eq":
        ok = abs(claim - value) <= (tol if tol is not None else 5e-4 * max(1, abs(claim)))
    elif kind == "round":  # claim is the rounded value at the given decimals
        ok = round(value, tol) == claim
    elif kind == "count":
        ok = claim == value
    elif kind == "text":
        ok = claim == value
    results.append((ok, label, claim, value))
    print(("PASS" if ok else "FAIL"), label, "| claimed", claim, "| computed", value if isinstance(value, str) else (round(value, 6) if isinstance(value, float) else value))


G = load_dir("grid"); GH = load_dir("grid_h"); B1 = load_dir("boot_primary"); B2 = load_dir("boot_primary_v2"); BM = load_dir("boot_meanerror_lit")
markets = sorted(G); st = np.array([m[0] for m in markets]); states = sorted(set(st))
K = json.load(open(os.path.join(R, "kappa.json"))); KW = json.load(open(os.path.join(R, "kappa_workers.json"))); K8 = json.load(open(os.path.join(R, "kappa_edcat8.json")))
PV = json.load(open(os.path.join(R, "piaac_validity.json")))["specs"]; PG = json.load(open(os.path.join(R, "pooled_gap.json"))); HY = json.load(open(os.path.join(R, "hypotheses.json")))
MS = json.load(open(os.path.join(R, "meanerror_summary.json"))); MW = json.load(open(os.path.join(R, "meanerror_summary_workers.json")))
SL = json.load(open(os.path.join(R, "schooling_lead.json"))); SL25 = json.load(open(os.path.join(R, "schooling_lead_age2565.json")))

print("== The reversal across 250 labour markets")
prim = [spec(G[m], "acs50k", "premarket", "logistic") for m in markets]
Rp = np.array([s["R"] for s in prim])
check("R positive in 250 primary", 250, int((Rp > 0).sum()), kind="count")
check("R min 0.117", 0.117, float(Rp.min()), 3, "round"); check("R max 0.261", 0.261, float(Rp.max()), 3, "round")
check("grid boot interval excludes zero in 250", 250, sum(1 for s in prim if s["boot"]["lo"] > 0), kind="count")
check("delta_merit>0 with joint-boot interval excluding zero in 246", 246, sum(1 for m in markets if B1[m]["intervals"]["delta_merit"]["lo"] > 0), kind="count")
allR = np.array([s["R"] for c in G.values() for s in c["spec_results"]]); allL = np.array([s["delta_label"] for c in G.values() for s in c["spec_results"]])
check("R positive in all 3000", 3000, int((allR > 0).sum()), kind="count"); check("R min over grid 0.042", 0.042, float(allR.min()), 3, "round"); check("R max over grid 0.280", 0.280, float(allR.max()), 3, "round")
check("delta_label negative in all 3000", 3000, int((allL < 0).sum()), kind="count")
check("mean R 0.171", 0.171, float(Rp.mean()), 3, "round"); check("sd R 0.027", 0.027, float(Rp.std(ddof=1)), 3, "round")
q75, q25 = np.percentile(Rp, [75, 25]); check("IQR 0.032", 0.032, float(q75 - q25), 3, "round")
pm = [s for c in G.values() for s in c["spec_results"] if s["covariate_set"] == "premarket"]
check("corr(R, -delta_label) premarket runs 0.957", 0.957, float(np.corrcoef([s["R"] for s in pm], [-s["delta_label"] for s in pm])[0, 1]), 3, "round")
rng = np.random.default_rng(20261001); by = {s: Rp[st == s] for s in states}
draws = np.array([np.concatenate([by[s] for s in rng.choice(states, 50, replace=True)]).mean() for _ in range(10000)])
lo, hi = np.percentile(draws, [2.5, 97.5]); check("state-block CI lo 0.164", 0.164, float(lo), 3, "round"); check("state-block CI hi 0.178", 0.178, float(hi), 3, "round")
check("width ratio about 2 (block se / iid se)", 2.0, float(draws.std(ddof=1) / (Rp.std(ddof=1) / math.sqrt(250))), 0.15)

print("== Reliability of the schooling proxy")
lw, nw = KW["lit"]["overall"], KW["num"]["overall"]; la, na = K["lit"]["overall"], K["num"]["overall"]
check("workers lit 0.243", 0.243, lw["kappa"], 3, "round"); check("workers lit CI lo 0.194", 0.194, lw["kappa"] - 1.96 * lw["se"], 3, "round"); check("workers lit CI hi 0.291", 0.291, lw["kappa"] + 1.96 * lw["se"], 3, "round")
check("workers n 1980", 1980, lw["n"], kind="count"); check("workers num 0.277", 0.277, nw["kappa"], 3, "round"); check("workers num CI 0.230", 0.230, nw["kappa"] - 1.96 * nw["se"], 3, "round"); check("workers num CI 0.324", 0.324, nw["kappa"] + 1.96 * nw["se"], 3, "round")
check("all lit 0.187", 0.187, la["kappa"], 3, "round"); check("all lit CI 0.152", 0.152, la["kappa"] - 1.96 * la["se"], 3, "round"); check("all lit CI 0.222", 0.222, la["kappa"] + 1.96 * la["se"], 3, "round"); check("all n 3758", 3758, la["n"], kind="count"); check("all num 0.226", 0.226, na["kappa"], 3, "round")
check("youngest band 0.03", 0.03, K["lit"]["age10"]["1"]["kappa"], 2, "round"); check("after 45 about 0.30 (bands 4,5 lit)", 0.30, float(np.mean([K["lit"]["age10"]["4"]["kappa"], K["lit"]["age10"]["5"]["kappa"]])), 0.01)
check("two codings agree within 0.008 (max |years-attainment| over lit/num)", True, max(abs(K[k]["overall"]["kappa"] - K8[k]["overall"]["kappa"]) for k in ("lit", "num")) < 0.008, kind="text")
print("   codings diff lit %.4f num %.4f" % (abs(K["lit"]["overall"]["kappa"] - K8["lit"]["overall"]["kappa"]), abs(K["num"]["overall"]["kappa"] - K8["num"]["overall"]["kappa"])))

print("== The correction under classical error")
U = np.array([s["decomp_unexplained"] for s in prim]); Uk = np.array([s["eiv_unexplained_at_kappa"] for s in prim])
check("uncorrected negative in 250", 250, int((U < 0).sum()), kind="count"); check("corrected negative in 250", 250, int((Uk < 0).sum()), kind="count")
check("pooled uncorrected -0.461", -0.461, PG["uncorrected_residual"]["mean"], 3, "round"); check("pooled corrected -0.630", -0.630, PG["corrected_residual"]["mean"], 3, "round"); check("deepening -0.170", -0.170, PG["deepening"]["mean"], 3, "round")
for k, lo_, hi_ in (("uncorrected_residual", -0.476, -0.446), ("corrected_residual", -0.651, -0.609), ("deepening", -0.188, -0.153)):
    check(f"{k} CI lo {lo_}", lo_, PG[k]["state_block_bootstrap"]["ci_lo"], 3, "round"); check(f"{k} CI hi {hi_}", hi_, PG[k]["state_block_bootstrap"]["ci_hi"], 3, "round")
check("LOSO corrected min -0.634", -0.634, PG["corrected_residual"]["loso"]["min"], 3, "round"); check("LOSO corrected max -0.627", -0.627, PG["corrected_residual"]["loso"]["max"], 3, "round")
check("each market's corrected interval excludes zero (250)", 250, sum(B1[m]["intervals"]["corrected_residual"]["hi"] < 0 for m in markets), kind="count")
w1 = np.mean([B2[m]["intervals"]["corrected_residual"]["hi"] - B2[m]["intervals"]["corrected_residual"]["lo"] for m in markets]); w2 = np.mean([B2[m]["intervals"]["corrected_residual_kappa_drawn"]["hi"] - B2[m]["intervals"]["corrected_residual_kappa_drawn"]["lo"] for m in markets])
check("kappa drawn widens by a third (ratio ~1.33)", 1.33, float(w2 / w1), 0.05); check("kappa drawn admits zero nowhere", 250, sum(B2[m]["intervals"]["corrected_residual_kappa_drawn"]["hi"] < 0 for m in markets), kind="count")
check("deepens in 247", 247, int((Uk < U).sum()), kind="count"); check("deepening interval excludes zero in 245", 245, sum(B1[m]["intervals"]["deepening"]["hi"] < 0 for m in markets), kind="count")
lead_code = {m: SL["markets"][f"{m[0]} {m[1]}"]["premarket"]["code"]["lead_women_minus_men"] for m in markets}
men = {m: v for m, v in lead_code.items() if v < 0}
check("men lead only in UT 2017-2019", "UT 2017, UT 2018, UT 2019", ", ".join(sorted(f"{m[0]} {m[1]}" for m in men)), kind="text")
check("men lead by at most 0.04 code points", 0.04, float(max(abs(v) for v in men.values())), 2, "round")
check("median lead elsewhere 0.32", 0.32, float(np.median([v for v in lead_code.values() if v > 0])), 2, "round")
neg_sweep = sum(1 for m in markets for cs in ("premarket", "extended") if all(k["unexplained"] < 0 for k in spec(G[m], "acs50k", cs, "logistic")["kappa_sweep"]))
check("corrected gap negative at every sweep value in all 500 decompositions", 500, neg_sweep, kind="count")
dml = np.array([s["dml_theta_logpts"] for s in prim])
check("DML corr with uncorrected residual 0.909", 0.909, float(np.corrcoef(dml, U)[0, 1]), 3, "round"); check("DML sign agrees in 250", 250, int((np.sign(dml) == np.sign(U)).sum()), kind="count")

print("== The proxy overstates women's assessed skill")
v = PV["lit"]["all"]["years_noage"]
check("women lead 0.367 (0.053)", 0.367, v["dxbar"]["point"], 3, "round"); check("lead se 0.053", 0.053, v["dxbar"]["se"], 3, "round")
check("benchmark +13.3 (2.2)", 13.3, v["benchmark"]["point"], 1, "round"); check("benchmark se 2.2", 2.2, v["benchmark"]["se"], 1, "round")
check("coefficient +0.10 (2.14)", 0.10, v["b_female"]["point"], 2, "round"); check("coef se 2.14", 2.14, v["b_female"]["se"], 2, "round")
check("fifth of a sd: benchmark / pv_sd about 0.2", 0.21, v["benchmark"]["point"] / v["pv_sd"]["point"], 0.02)
check("difference -13.2 (3.3)", -13.2, v["difference"]["point"], 1, "round"); check("difference se 3.3", 3.3, v["difference"]["se"], 1, "round"); check("z -3.97", -3.97, v["difference"]["z"], 2, "round")
n = PV["num"]["all"]["years_noage"]
check("numeracy coef -11.9 (2.3)", -11.9, n["b_female"]["point"], 1, "round"); check("num coef se 2.3", 2.3, n["b_female"]["se"], 1, "round"); check("num benchmark +12.1 (2.0)", 12.1, n["benchmark"]["point"], 1, "round"); check("num benchmark se 2.0", 2.0, n["benchmark"]["se"], 1, "round")
check("num difference -24.0 (3.2)", -24.0, n["difference"]["point"], 1, "round"); check("num diff se 3.2", 3.2, n["difference"]["se"], 1, "round"); check("num z -7.47", -7.47, n["difference"]["z"], 2, "round")
check("delta lit 0.297 (0.069)", 0.297, v["delta_proxy_units"]["point"], 3, "round"); check("delta lit se 0.069", 0.069, v["delta_proxy_units"]["se"], 3, "round")
check("delta num 0.560 (0.067)", 0.560, n["delta_proxy_units"]["point"], 3, "round"); check("delta num se 0.067", 0.067, n["delta_proxy_units"]["se"], 3, "round")
check("workers lit 0.580 (0.136)", 0.580, PV["lit"]["workers"]["years_noage"]["delta_proxy_units"]["point"], 3, "round"); check("workers lit se 0.136", 0.136, PV["lit"]["workers"]["years_noage"]["delta_proxy_units"]["se"], 3, "round")
check("workers num 0.871 (0.131)", 0.871, PV["num"]["workers"]["years_noage"]["delta_proxy_units"]["point"], 3, "round"); check("workers num se 0.131", 0.131, PV["num"]["workers"]["years_noage"]["delta_proxy_units"]["se"], 3, "round")
zs = [PV[s][m][sp]["difference"]["z"] for s in ("lit", "num") for m in ("all", "workers") for sp in ("years_noage", "years_age")]
check("z between -3.97 and -7.47 (max)", -3.97, max(zs), 2, "round"); check("z min -7.47", -7.47, min(zs), 2, "round")
check("age bands move delta by at most 0.02", True, max(abs(PV[s][m]["years_age"]["delta_proxy_units"]["point"] - PV[s][m]["years_noage"]["delta_proxy_units"]["point"]) for s in ("lit", "num") for m in ("all", "workers")) <= 0.02, kind="text")
check("44 replicate weights", 44, v["n_replicates"], kind="count")

print("== The gap under the estimated error structure")
P = lambda summ, tag, cs, key: summ["arms"][tag]["covsets"][cs]["pooled"][key]
check("uncorrected years-coded -0.457", -0.457, P(MS, "lit", "premarket", "U_y1")["mean"], 3, "round"); check("classical corrected -0.609", -0.609, P(MS, "lit", "premarket", "U_yk")["mean"], 3, "round")
for tag, val, lo_, hi_ in (("lit", -0.399, -0.417, -0.381), ("num", -0.212, -0.231, -0.193), ("num_hi", -0.112, -0.132, -0.093)):
    p = P(MS, tag, "premarket", "U_mk"); check(f"{tag} premarket {val}", val, p["mean"], 3, "round"); check(f"{tag} lo {lo_}", lo_, p["ci_lo"], 3, "round"); check(f"{tag} hi {hi_}", hi_, p["ci_hi"], 3, "round")
check("num_hi delta 0.694", 0.694, MS["arms"]["num_hi"]["delta_years"], 3, "round")
check("adverse in 250 (lit)", 250, MS["arms"]["lit"]["covsets"]["premarket"]["U_m_anchor_negative"], kind="count"); check("adverse in 250 (num)", 250, MS["arms"]["num"]["covsets"]["premarket"]["U_m_anchor_negative"], kind="count"); check("adverse in 235 (num_hi)", 235, MS["arms"]["num_hi"]["covsets"]["premarket"]["U_m_anchor_negative"], kind="count")
for tag, val, lo_, hi_ in (("lit_workers", -0.353, -0.371, -0.336), ("num_workers", -0.198, -0.217, -0.181)):
    p = P(MW, tag, "premarket", "U_mk"); check(f"{tag} {val}", val, p["mean"], 3, "round"); check(f"{tag} lo {lo_}", lo_, p["ci_lo"], 3, "round"); check(f"{tag} hi {hi_}", hi_, p["ci_hi"], 3, "round")
    check(f"{tag} adverse in 250", 250, MW["arms"][tag]["covsets"]["premarket"]["U_m_anchor_negative"], kind="count")
check("workers anchor 0.243", 0.243, json.load(open(os.path.join(R, "grid_years_age2565", "AK_2017.json")))["spec_results"][0]["kappa_point"], 3, "round")
check("both drawn: interval below zero in 250", 250, sum(BM[m]["intervals"]["corrected_residual_both_drawn"]["hi"] < 0 for m in markets), kind="count")
check("indistinguishable from uncorrected in 190", 190, sum(BM[m]["intervals"]["net_vs_reported_both_drawn"]["lo"] < 0 < BM[m]["intervals"]["net_vs_reported_both_drawn"]["hi"] for m in markets), kind="count")
def fit(summ, cs):
    arms = sorted(summ["arms"].items(), key=lambda kv: kv[1]["delta_years"]); first = arms[0][1]["covsets"][cs]
    xs = [0.0] + [a["delta_years"] for _, a in arms]; ys = [first["pooled"]["U_yk"]["mean"]] + [a["covsets"][cs]["pooled"]["U_mk"]["mean"] for _, a in arms]
    b, a0 = np.polyfit(xs, ys, 1); per = {mk["market"]: [mk["U_yk"]] for mk in first["markets"]}
    for _, a in arms:
        for mk in a["covsets"][cs]["markets"]: per[mk["market"]].append(mk["U_mk"])
    pmz = np.array([-np.polyfit(xs, v, 1)[1] / np.polyfit(xs, v, 1)[0] for v in per.values()])
    return a0, b, max(abs(a0 + b * x - y) for x, y in zip(xs, ys)), pmz, first["pooled"]["U_y1"]["mean"]
a0, b, res, pmz, U1 = fit(MS, "premarket"); a0w, bw, resw, pmzw, U1w = fit(MW, "premarket")
check("fit intercept -0.610", -0.610, a0, 3, "round"); check("fit slope 0.714", 0.714, b, 3, "round"); check("workers intercept -0.661", -0.661, a0w, 3, "round"); check("workers slope 0.531", 0.531, bw, 3, "round")
check("largest residual 0.003", 0.003, max(res, resw), 3, "round")
# corrected return relative to uncorrected: the slope is b(kappa); b(1) is not stored; check slope*kappa against the 1/kappa claim loosely
check("slope is 5.6 x uncorrected return (b(kappa)*kappa ~ b(1)?): slope*kappa", 0.133, b * ANCHOR, 3, "round")
check("neutral at 0.214", 0.214, (U1 - a0) / b, 3, "round"); check("workers neutral 0.368", 0.368, (U1w - a0w) / bw, 3, "round")
check("gap closes at 0.855", 0.855, -a0 / b, 3, "round"); check("workers closes at 1.245", 1.245, -a0w / bw, 3, "round")
check("per-market closing min 0.635", 0.635, float(pmz.min()), 3, "round"); check("per-market closing max 1.356", 1.356, float(pmz.max()), 3, "round")
check("above numeracy estimate in 250", 250, int((pmz > 0.560).sum()), kind="count"); check("above upper bound in all but twelve (238)", 238, int((pmz > 0.694).sum()), kind="count")
check("women lead 0.247 years (all, premarket)", 0.247, SL["summary"]["premarket_years"]["mean"], 3, "round"); check("women lead 0.469 (earners)", 0.469, SL25["summary"]["premarket_years"]["mean"], 3, "round")
rule = []
for summ, sl_ in ((MS, SL), (MW, SL25)):
    for tag, arm in summ["arms"].items():
        d = arm["delta_years"]; ok = 0
        for mk in arm["covsets"]["premarket"]["markets"]:
            lead = sl_["markets"][mk["market"]]["premarket"]["years"]["lead_women_minus_men"]; ok += ((lead - d) > 0) == (mk["deep_m"] < 0)
        rule.append(ok)
check("sign rule matches 236 to 250 on premarket (min)", 236, min(rule), kind="count"); check("sign rule max 250", 250, max(rule), kind="count")
for tag, val, lo_, hi_ in (("lit", -0.256, -0.274, -0.239), ("num", -0.047, -0.064, -0.029), ("num_hi", 0.063, 0.044, 0.082)):
    p = P(MS, tag, "extended", "U_mk"); check(f"extended {tag} {val}", val, p["mean"], 3, "round"); check(f"extended {tag} lo {lo_}", lo_, p["ci_lo"], 3, "round"); check(f"extended {tag} hi {hi_}", hi_, p["ci_hi"], 3, "round")
check("extended num_hi adverse in 58", 58, MS["arms"]["num_hi"]["covsets"]["extended"]["U_m_anchor_negative"], kind="count")
p = P(MW, "num_workers", "extended", "U_mk"); check("earners extended numeracy +0.015", 0.015, p["mean"], 3, "round"); check("lo +0.001", 0.001, p["ci_lo"], 3, "round"); check("hi +0.029", 0.029, p["ci_hi"], 3, "round")
# "no combination of a reliability in the swept range and an error difference the survey supports closes the gap in any market" (primary set, deltas up to 0.560 and the earners deltas on their sample)
flips = sum(MS["arms"][t]["covsets"]["premarket"]["U_m_sign_flips_in_sweep"] for t in ("lit_lo", "lit", "lit_hi", "num")) + sum(MW["arms"][t]["covsets"]["premarket"]["U_m_sign_flips_in_sweep"] for t in ("lit_workers", "num_workers"))
check("sweep crossings on the primary set: 6 (num, all) and 78 (num, earners), none elsewhere at survey estimates", 84, flips, kind="count")
print("   note: at the upper numeracy bound 0.694 the premarket sweep crosses zero in", MS["arms"]["num_hi"]["covsets"]["premarket"]["U_m_sign_flips_in_sweep"], "markets")

print("== The differential-reliability channel")
diffs = [K[k]["sex_difference"] for k in ("lit", "num")] + [KW[k]["sex_difference"] for k in ("lit", "num")] + [K8[k]["sex_difference"] for k in ("lit", "num")]
check("differential min -0.002", -0.002, min(d["difference"] for d in diffs), 3, "round"); check("differential max +0.034", 0.034, max(d["difference"] for d in diffs), 3, "round")
check("se min 0.030", 0.030, min(d["se"] for d in diffs), 3, "round"); check("se max 0.044", 0.044, max(d["se"] for d in diffs), 3, "round")
kf, km = K["lit"]["sex"]["2"]["kappa"], K["lit"]["sex"]["1"]["kappa"]; ks_, need = [], []
for m in markets:
    g = B1[m]["group_moments"]["SCHL"]; pf = B1[m]["group_weight_share_female"]; pm_ = 1 - pf
    Vf, Vm, muf, mum = g["female"]["var"], g["male"]["var"], g["female"]["mean"], g["male"]["mean"]
    V = pf * Vf + pm_ * Vm + pf * pm_ * (muf - mum) ** 2
    ks_.append(1 - (pf * (1 - kf) * Vf + pm_ * (1 - km) * Vm) / V)
    kbar = 0.5 * (kf + km); sens = -(pf * Vf - pm_ * Vm) / (2 * V); k0 = 1 - (pf * (1 - kbar) * Vf + pm_ * (1 - kbar) * Vm) / V
    need.append(abs((0.15 - k0) / sens))
check("kstar min 0.186", 0.186, float(min(ks_)), 3, "round"); check("kstar max 0.204", 0.204, float(max(ks_)), 3, "round")
gap_at = [np.interp(k, [x["kappa"] for x in s["kappa_sweep"]], [x["unexplained"] for x in s["kappa_sweep"]]) for k, s in zip(ks_, prim)]
check("corrected gap at kstar more than 0.4 below zero (max)", True, max(gap_at) < -0.4, kind="text"); print("   max gap at kstar %.4f" % max(gap_at))
check("differential needed at least 0.435", 0.435, float(min(need)), 3, "round"); check("arithmetic maximum 0.373", 0.373, float(kf + km), 3, "round")

print("== Specification dependence")
strict = lambda s: s["delta_merit"] > 0 > s["delta_label"]
allspecs = [s for c in G.values() for s in c["spec_results"]]
check("premarket own sample 95.1", 95.1, 100 * np.mean([strict(s) for s in allspecs if s["covariate_set"] == "premarket"]), 1, "round")
check("extended 0.1", 0.1, 100 * np.mean([strict(s) for s in allspecs if s["covariate_set"] == "extended"]), 1, "round")
hs = [s for c in GH.values() for s in c["spec_results"]]
check("harmonised premarket 98.2", 98.2, 100 * np.mean([strict(s) for s in hs if s["covariate_set"] == "premarket_h"]), 1, "round")
check("decline 95.0", 95.0, 100 * (np.mean([strict(s) for s in allspecs if s["covariate_set"] == "premarket"]) - np.mean([strict(s) for s in allspecs if s["covariate_set"] == "extended"])), 1, "round")
check("sample contribution -3.1", -3.1, 100 * (np.mean([strict(s) for s in allspecs if s["covariate_set"] == "premarket"]) - np.mean([strict(s) for s in hs if s["covariate_set"] == "premarket_h"])), 1, "round")
check("conditioning contribution +98.1", 98.1, 100 * (np.mean([strict(s) for s in hs if s["covariate_set"] == "premarket_h"]) - np.mean([strict(s) for s in allspecs if s["covariate_set"] == "extended"])), 1, "round")
for l, val in (("logistic", 99.0), ("tabpfn", 98.0), ("xgboost", 97.6)):
    check(f"harmonised {l} {val}", val, 100 * np.mean([strict(s) for s in hs if s["covariate_set"] == "premarket_h" and s["learner"] == l]), 1, "round")
# largest spread across learners in any comparison: within each (arm, covariate set, outcome) block
spreads = {}
for name, cells in (("grid", G), ("grid_h", GH)):
    for cs in ("premarket", "extended", "premarket_h"):
        for oc in ("acs50k", "cellmedian", None):
            rates = []
            for l in ("logistic", "xgboost", "tabpfn"):
                v = [strict(s) for c in cells.values() for s in c["spec_results"] if s["covariate_set"] == cs and s["learner"] == l and (oc is None or s["outcome_def"] == oc)]
                if v: rates.append(100 * np.mean(v))
            if rates: spreads[(name, cs, oc)] = max(rates) - min(rates)
big = max(spreads, key=spreads.get); check("largest learner spread 10.8", 10.8, spreads[big], 1, "round"); print("   at", big)
check("1.4 point spread on harmonised sample (all outcomes)", 1.4, spreads[("grid_h", "premarket_h", None)], 1, "round")
check("Gamma 0.012", 0.012, HY["H2_5"]["Gamma"], 3, "round"); check("omega share 0.008", 0.008, HY["H2_5"]["share_within_omega"], 3, "round")
half = sum(1 for m in markets if abs(np.mean([strict(s) for s in G[m]["spec_results"]]) - 0.5) < 1e-9)
print("   markets with strict share exactly one half:", half, "; within 0.05 of one half:", sum(1 for m in markets if abs(np.mean([strict(s) for s in G[m]["spec_results"]]) - 0.5) <= 0.05 + 1e-9))
check("agreement pinned near one half in 217 markets", 217, half, kind="count")
# five further arms
def arm(d, cs="premarket", oc="acs50k"):
    cells = load_dir(d); return cells, [spec(cells[m], oc, cs, "logistic") for m in markets]
W, ws = arm("grid_wage"); _, wse = arm("grid_wage", "extended")
check("wage halves the raw gap: ratio", 0.55, np.mean([s["decomp_raw_logpts"] for s in ws]) / np.mean([s["decomp_raw_logpts"] for s in prim]), 0.06)
check("wage corrected near -0.52 (premarket)", -0.52, float(np.mean([s["eiv_unexplained_at_kappa"] for s in ws])), 2, "round"); check("wage corrected near -0.52 (extended)", -0.52, float(np.mean([s["eiv_unexplained_at_kappa"] for s in wse])), 2, "round")
A, as_ = arm("grid_adjinc"); _, ase = arm("grid_adjinc", "extended")
check("adjinc log decomposition invariant (max change < 1e-9)", True, max(abs(a["decomp_unexplained"] - p["decomp_unexplained"]) for a, p in zip(as_, prim)) < 1e-9, kind="text")
ext = [spec(G[m], "acs50k", "extended", "logistic") for m in markets]
check("adjinc moves R by at most 0.027 (ceiling)", True, float(max(max(abs(a["R"] - p["R"]) for a, p in zip(as_, prim)), max(abs(a["R"] - p["R"]) for a, p in zip(ase, ext)))) <= 0.027, kind="text")
Wm, wm = arm("grid_wmedian", "premarket", "cellmedian"); _, wme = arm("grid_wmedian", "extended", "cellmedian")
cm = [spec(G[m], "cellmedian", "premarket", "logistic") for m in markets]; cme = [spec(G[m], "cellmedian", "extended", "logistic") for m in markets]
check("wmedian changes no market's sign", 0, sum(np.sign(a["R"]) != np.sign(p["R"]) for a, p in list(zip(wm, cm)) + list(zip(wme, cme))), kind="count")
Ag, ag = arm("grid_age2565"); check("age2565 deepening -0.201", -0.201, float(np.mean([s["eiv_unexplained_at_kappa"] - s["decomp_unexplained"] for s in ag])), 3, "round")
for d in ("grid_wage", "grid_adjinc", "grid_wmedian", "grid_age2565"):
    cells = load_dir(d); check(f"{d}: R positive in all 250 (premarket and extended)", 500, sum(1 for m in markets for cs in ("premarket", "extended") if spec(cells[m], "acs50k", cs, "logistic")["R"] > 0), kind="count")
O, oc_ = arm("grid_occ", "extended_occ")
check("occupation: R positive in 250", 250, sum(s["R"] > 0 for s in oc_), kind="count"); check("occupation mean R 0.077", 0.077, float(np.mean([s["R"] for s in oc_])), 3, "round"); check("extended mean R without occupation 0.113", 0.113, float(np.mean([s["R"] for s in ext])), 3, "round")
thr = np.array([s["eiv_min_kappa"] for s in oc_]); check("threshold averages 0.254", 0.254, float(thr.mean()), 3, "round"); check("threshold exceeds anchor in every market", 250, int((thr > ANCHOR).sum()), kind="count")
at35 = [next(k["unexplained"] for k in s["kappa_sweep"] if abs(k["kappa"] - 0.35) < 1e-9) for s in oc_]
check("occ gap -0.168 at 0.35", -0.168, float(np.mean(at35)), 3, "round"); check("negative in 249 of 250 at 0.35", 249, sum(v < 0 for v in at35), kind="count")
ratios = np.array([s["delta_merit"] / s["delta_label"] for s in allspecs if s["covariate_set"] == "extended"])
check("extended absorb median 32 per cent", 32, round(100 * float(np.median(ratios))), kind="count")

print("== Introduction and abstract")
check("0.19 lit all", 0.19, la["kappa"], 2, "round"); check("0.23 num all", 0.23, na["kappa"], 2, "round"); check("0.24 lit earners", 0.24, lw["kappa"], 2, "round"); check("0.28 num earners", 0.28, nw["kappa"], 2, "round")
check("-0.46 to -0.63", -0.46, PG["uncorrected_residual"]["mean"], 2, "round"); check("-0.63", -0.63, PG["corrected_residual"]["mean"], 2, "round")
check("-0.40 lit", -0.40, P(MS, "lit", "premarket", "U_mk")["mean"], 2, "round"); check("-0.21 num", -0.21, P(MS, "num", "premarket", "U_mk")["mean"], 2, "round")
check("closing is about 2.9 times the literacy estimate", 2.9, (-a0 / b) / PV["lit"]["all"]["years_noage"]["delta_proxy_units"]["point"], 0.1)
print("   ratios: full-sample closing / lit 0.297 = %.2f; / num 0.560 = %.2f; earners closing / lit 0.580 = %.2f; / num 0.871 = %.2f" % ((-a0 / b) / 0.297, (-a0 / b) / 0.560, (-a0w / bw) / 0.580, (-a0w / bw) / 0.871))
check("(1-kappa)/kappa about 4.4 at the anchor", 4.4, (1 - ANCHOR) / ANCHOR, 1, "round")
check("women lead in 247 of 250 (code units)", 247, sum(1 for v in lead_code.values() if v > 0), kind="count")
check("0.9 years closing (Discussion) full sample", 0.9, -a0 / b, 1, "round"); check("1.2 among earners", 1.2, -a0w / bw, 1, "round")
print("   estimates between 0.3 and 0.9: min delta %.3f max %.3f" % (min(PV[s][m]["years_noage"]["delta_proxy_units"]["point"] for s in ("lit", "num") for m in ("all", "workers")), max(PV[s][m]["years_noage"]["delta_proxy_units"]["point"] for s in ("lit", "num") for m in ("all", "workers"))))

fails = [r for r in results if not r[0]]
print(f"\n{len(results)} checks, {len(fails)} failed")
for f in fails:
    print("  FAIL:", f[1], "claimed", f[2], "computed", f[3])
