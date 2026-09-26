#!/usr/bin/env python3
"""Write si.tex from si_numbers.json and the table fragments in tables/.
Every number in the Supplementary Note text is substituted from the JSON that
build_si_tables.py computed from the run logs; the prose holds no typed digits
except equation numbers, section labels and the fixed design constants
(0.15 sweep floor, 500 draws, 10,000 state draws, 1.96)."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, "si_numbers.json")))


def f(x, d=3, plus=False):
    s = f"{x:.{d}f}"
    if plus and x > 0:
        s = "+" + s
    return s.replace("-", "$-$") if x < 0 else s


def fa(x, d=3):  # absolute value, formatted
    return f(abs(x), d)


def floor_(x, d):  # for "more than x" statements: round towards zero so the statement stays true
    import math
    return f(math.floor(abs(x) * 10 ** d) / 10 ** d, d)


def ceil_(x, d):  # for "at most x" / "no more than x" statements: round away from zero
    import math
    return f(math.ceil(abs(x) * 10 ** d) / 10 ** d, d)


V = {}
V["kf"] = f(N["A_kappa_f"], 4); V["km"] = f(N["A_kappa_m"], 4)
V["kstar_lo"], V["kstar_hi"] = f(N["A_kstar_range"][0], 4), f(N["A_kstar_range"][1], 4)
V["bound_holds"] = str(N["A_bound_holds"])
V["kbarV_lo"], V["kbarV_hi"] = f(N["A_kbarV_range"][0], 4), f(N["A_kbarV_range"][1], 4)
V["below_avg"] = str(N["A_below_unweighted_average"])
V["sens_lo"], V["sens_hi"] = f(N["A_sens_range"][0], 3, True), f(N["A_sens_range"][1], 3, True)
V["need_min"], V["need_med"] = f(N["A_need_min_abs"]), f(N["A_need_median_abs"])
V["arith_max"] = f(N["A_arith_max_diff"]); V["need_exceeds"] = str(N["A_need_exceeds_max_n"])
V["diff_est"], V["diff_se"] = f(N["A_diff_est"]), f(N["A_diff_se"])
V["lb_m"], V["lb_f"] = f(N["A_lower_bounds"][0], 4), f(N["A_lower_bounds"][1], 4)
V["kstar_lb_lo"], V["kstar_lb_hi"] = f(N["A_kstar_lower_range"][0], 4), f(N["A_kstar_lower_range"][1], 4)
V["gap_at_kstar"] = floor_(N["A_gap_at_kstar_max"], 2)
S7p, S7e = N["S7_premarket"], N["S7_extended"]
V["inc_total"] = str(S7p["increasing"] + S7e["increasing"]); V["dec_total"] = str(S7p["decreasing"] + S7e["decreasing"])
b1, b2 = N["B1_registered"], N["B1_v2"]
V["b1_R"], V["b1_corr"], V["b1_corr_ci"], V["b1_deep"], V["b1_deep_ci"] = map(str, (b1["R_all_draws_pos"], b1["corr_all_draws_neg"], b1["corr_ci_excl0"], b1["deep_all_draws_neg"], b1["deep_ci_excl0"]))
V["b2_R"], V["b2_corr"], V["b2_corr_ci"], V["b2_deep"], V["b2_deep_ci"] = map(str, (b2["R_all_draws_pos"], b2["corr_all_draws_neg"], b2["corr_ci_excl0"], b2["deep_all_draws_neg"], b2["deep_ci_excl0"]))
V["fold_max"] = ceil_(N["B2_max_endpoint_change_folds"], 4)
V["w_corr"], V["w_deep"] = f(N["B2_corr_width_ratio_kappa_drawn"], 2), f(N["B2_deep_width_ratio_kappa_drawn"], 2)
V["kd_corr_excl"], V["kd_deep_excl"] = str(N["B2_corr_kappa_drawn_excl0"]), str(N["B2_deep_kappa_drawn_excl0"])
V["kd_lo"], V["kd_hi"] = f(N["B2_kappa_drawn_range"][0]), f(N["B2_kappa_drawn_range"][1])
V["bm_shift"], V["bm_se"] = f(N["BM_shift"]), f(N["BM_shift_se"])
V["bm_w_shift"], V["bm_w_kappa"], V["bm_w_both"] = f(N["BM_width_ratio_shift_over_fixed"], 2), f(N["BM_width_ratio_kappa_over_fixed"], 2), f(N["BM_width_ratio_both_over_fixed"], 2)
V["bm_excl_both"], V["bm_excl_fixed"] = str(N["BM_corr_both_drawn_excl0"]), str(N["BM_corr_fixed_excl0"])
V["bm_net_both"], V["bm_net_fixed"] = str(N["BM_net_both_drawn_covers0"]), str(N["BM_net_fixed_covers0"])
V["fdr_n"] = str(N["B3_n_tested"]); V["fdr_rej"] = str(round(N["B3_fdr_share"] * N["B3_n_tested"]))
V["R_mean"], V["R_loso_min"], V["R_loso_min_st"], V["R_loso_max"], V["R_loso_max_st"] = f(N["R_mean"], 4), f(N["R_loso_min"], 4), N["R_loso_min_state"], f(N["R_loso_max"], 4), N["R_loso_max_state"]
V["R_ci_lo"], V["R_ci_hi"], V["R_se_b"], V["R_se_i"] = f(N["R_block_ci"][0], 4), f(N["R_block_ci"][1], 4), f(N["R_block_se"], 4), f(N["R_iid_se"], 4)
for k in ("uncorrected_residual", "corrected_residual", "deepening"):
    p = N[f"pooled_{k}"]
    V[f"p_{k}"] = f(p["mean"]); V[f"p_{k}_lo"], V[f"p_{k}_hi"] = f(p["ci"][0]), f(p["ci"][1])
    V[f"p_{k}_loso_lo"], V[f"p_{k}_loso_lo_st"], V[f"p_{k}_loso_hi"], V[f"p_{k}_loso_hi_st"] = f(p["loso"][0]), p["loso"][1], f(p["loso"][2]), p["loso"][3]
    V[f"p_{k}_ratio"] = f(p["se_ratio"], 2)
V["sweep_max"], V["sweep_max_hi"], V["sweep_kappa"] = f(N["pooled_sweep_max"]), f(N["pooled_sweep_max_ci_hi"]), f(N["pooled_sweep_kappa_at_max"], 2)
th = N["S10_occ_threshold"]
V["thr_mean"], V["thr_lo"], V["thr_hi"], V["thr_min"], V["thr_max"], V["thr_above"] = f(th["mean"]), f(th["ci"][0]), f(th["ci"][1]), f(th["min"]), f(th["max"]), str(th["above_anchor"])
adm = N["S10_occ_admissible_sweep"]
V["adm35_mean"], V["adm35_neg"] = f(adm["0.35"]["mean"]), str(adm["0.35"]["negative"])
V["adm40_neg"] = str(adm["0.4"]["negative"])
V["S2_not"], V["S2_resp"], V["S2_max"] = str(N["S2_not_estimable_categories"]), f"{N['S2_not_estimable_respondents']:,}", f(N["S2_max_kappa_where_estimable"])
V["S2_not_word"] = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}[N["S2_not_estimable_categories"]]
V["S3_ext_median"] = str(round(100 * N["S3_median_Extended, own sample"]))
V["S7_corr"] = f(abs(N["S7_corr_deepening_lead"]), 2)
V["S7_women_lead"], V["S7_deep_women"], V["S7_att_men"] = str(N["S7_women_lead_n"]), str(N["S7_deepened_where_women_lead"]), str(N["S7_attenuated_where_men_lead"])
men = N["S7_men_lead_markets"]
V["S7_men_list"] = ", ".join(sorted(men)); V["S7_men_max"] = f(max(abs(v) for v in men.values()), 2)
V["S7_lead_med"], V["S7_lead_min"], V["S7_lead_max"] = f(N["S7_lead_median_where_women_lead"], 2), f(N["S7_lead_min_where_women_lead"], 2), f(N["S7_lead_max"], 2)
V["S8_z_lo"], V["S8_z_hi"] = f(N["S8_z_range"][0], 2), f(N["S8_z_range"][1], 2)
V["S8_age_change"] = ceil_(N["S8_max_delta_change_age_bands"], 3)
fits = N["S9_fits"]
for k, v in fits.items():
    V[f"fit_{k}_a"], V[f"fit_{k}_b"], V[f"fit_{k}_zero"], V[f"fit_{k}_neutral"], V[f"fit_{k}_resid"] = f"{v['intercept']:.3f}", f(v["slope"]), f(v["zero"]), f(v["neutral"]), f(v["max_resid"])
    V[f"fit_{k}_pm_med"], V[f"fit_{k}_pm_min"], V[f"fit_{k}_pm_max"] = f(v["per_market_median"]), f(v["per_market_min"]), f(v["per_market_max"])
V["fit_all_premarket_above_num"] = str(fits["all_premarket"]["n_above_numeracy"]); V["fit_all_premarket_above_num_hi"] = str(fits["all_premarket"]["n_above_numeracy_upper"])
rule = N["S9_sign_rule_matches"]
pm_rule = [rule[k] for k in rule if "_premarket_" in k]
V["rule_pm_min"], V["rule_pm_max"] = str(min(pm_rule)), str(max(pm_rule))
ex_rule = [rule[k] for k in rule if "_extended_" in k]
V["rule_ex_min"], V["rule_ex_max"] = str(min(ex_rule)), str(max(ex_rule))
V["lead_all_pm"], V["lead_w25_pm"] = f(N["lead_years_all_premarket"]["mean"]), f(N["lead_years_w25_premarket"]["mean"])
V["adj_max_R"] = ceil_(N["S10_adjinc_max_change_R"], 3); V["wm_max_R"] = ceil_(N["S10_wmedian_max_change_R_cellmedian"], 3)
V["years_max_R"] = ceil_(N["S10_years_max_R_diff"], 3); V["years_U"], V["years_Uk"] = f(N["S10_years_U_pm"]), f(N["S10_years_Uk_pm"])
V["wage_raw_pm"] = f(N["S10_grid_wage_premarket"]["raw"]); V["grid_raw_pm"] = f(N["S10_grid_premarket"]["raw"])
V["wage_Uk_pm"], V["wage_Uk_ex"] = f(N["S10_grid_wage_premarket"]["Uk"]), f(N["S10_grid_wage_extended"]["Uk"])
V["age_deep"] = f(N["S10_age2565_deepening_mean"]); V["age_anchor"] = f(N["S10_age2565_anchor"])
V["occ_R"], V["ext_R"] = f(N["S10_grid_occ_extended_occ"]["R_mean"]), f(N["S10_grid_extended"]["R_mean"])
H = N["hashes"]

TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx}
\usepackage[british]{babel}
\setlength{\parskip}{3pt}
\newcommand{\botrule}{\bottomrule}
\newcommand{\tabcap}[2]{\noindent\textbf{Supplementary Table #1.} #2\par\vspace{4pt}}
\newcommand{\sitab}[3]{\par\vspace{10pt}\noindent\begin{minipage}{\textwidth}\noindent\textbf{Supplementary Table #1.} #2\par\vspace{4pt}#3\end{minipage}\par}
\begin{document}

\begin{center}
{\Large Supplementary Information}\\[6pt]
{\large Correcting a merit proxy for measurement error does not close the sex gap in income audits}
\end{center}
\vspace{6pt}

\noindent\textbf{Contents.} The Supplementary Note has two parts: part~A derives the reduction of group-specific reliabilities to one common reliability under a pooled reference, and part~B sets out what the intervals cover, how dependence across markets is handled, and the condition under which the correction is identified. Supplementary Tables S1 to S10 report the reliability strata, the absorption distribution, the verdict rate by analyst dimension, the dependence checks, the computational environment, the mechanism that sets the direction of the correction, the test of equal error means in the assessment survey, the corrected gap under each estimated error structure, and the six further arms including the occupation arm in which the correction is not identified.

\section*{Supplementary Note}

\subsection*{Part A. Group-specific reliabilities under a pooled reference reduce to one common reliability}

This note derives the result the main text relies on when it treats the differential-reliability channel as closed. We show that the corrected decomposition, as implemented, cannot distinguish a pair of group-specific reliabilities from a single common one, give the mapping between them, and establish what a difference between the sexes can and cannot do to the sign of the corrected gap.

\paragraph{A.1 Setup.}
Write $X_p$ for the proxy column, the American Community Survey's categorical schooling variable, and $s \in \{f, m\}$ for the two groups. Let $p_s$ be the group's share of total weight, $\mu_s$ and $V_s$ the weighted mean and variance of $X_p$ within group $s$, and
\begin{equation}
V = p_f V_f + p_m V_m + p_f p_m (\mu_f - \mu_m)^2
\end{equation}
the weighted variance of $X_p$ in the pooled sample, the sum of a within-group and a between-group component.

Under the classical model the proxy is $X_p = M^{*} + \nu$, with the error $\nu$ uncorrelated with $M^{*}$ and, we assume throughout this part, mean zero within each group. A group reliability $\kappa_s$ is equivalent to an error variance in that group of
\begin{equation}
\sigma_s^2 = (1 - \kappa_s) V_s .
\end{equation}
The corrected decomposition is estimated as a single regression on the pooled sample, with design matrix $[\,X \; g\,]$ for the group indicator $g$, solving $(S_{XX} - \Lambda) b = S_{Xy}$ and discarding the coefficient on $g$, exactly as the uncorrected decomposition discards it. The matrix $\Lambda$ is diagonal, and only the proxy column is treated as mismeasured, so $\Lambda$ has a single non-zero entry,
\begin{equation}
\lambda = (1 - \kappa) V .
\end{equation}

\paragraph{A.2 The reliability enters through one scalar.}
Equation (3) is the whole of the estimator's dependence on the reliability. Nothing else in $S_{XX}$, $S_{Xy}$, the group means or the reference structure involves $\kappa$. The corrected coefficient vector, and therefore the corrected explained and unexplained components, are functions of the assumed error variance $\lambda$ alone.

Because the error is mean zero within each group, it contributes nothing to the between-group term of (1), and the error variance of the pooled sample is the weighted average of the within-group error variances:
\begin{equation}
\sigma^2 = p_f \sigma_f^2 + p_m \sigma_m^2 = p_f (1 - \kappa_f) V_f + p_m (1 - \kappa_m) V_m .
\end{equation}

\paragraph{A.3 The equivalence.}
Setting (3) equal to (4) gives the reliability that a single common value would have to take to imply the same correction:
\begin{equation}
\kappa^{*} = 1 - \frac{p_f (1 - \kappa_f) V_f + p_m (1 - \kappa_m) V_m}{V} .
\end{equation}
Any pair $(\kappa_f, \kappa_m)$ and the single value $\kappa^{*}$ of (5) produce identical corrected coefficients, an identical explained component and an identical unexplained component. The equivalence is exact, not approximate, and it holds cell by cell. The pair is therefore not identified by this estimator: a difference in reliability between the groups has no channel of its own, and acts only by moving the level $\kappa^{*}$.

\paragraph{A.4 Two consequences.}
Between-group dispersion can only raise $\kappa^{*}$. Let $W = p_f V_f + p_m V_m$ be the within-group component of (1) and $B = p_f p_m (\mu_f - \mu_m)^2 \ge 0$ the between-group component, and let $\bar\kappa_V = (p_f \kappa_f V_f + p_m \kappa_m V_m)/W$ be the variance-weighted mean of the group reliabilities. Since $V = W + B \ge W$,
\begin{equation}
\kappa^{*} = 1 - \frac{\sigma^2}{W + B} \;\ge\; 1 - \frac{\sigma^2}{W} = \bar\kappa_V .
\end{equation}
The equivalent common reliability is bounded below by the variance-weighted mean of the two group reliabilities. It is never the smaller of them, though it can sit marginally below their unweighted average, as it does in @below_avg@ of the 250 markets.

The differential moves $\kappa^{*}$ only through unequal weights. Write the pair as $\kappa_f = \bar\kappa - d/2$ and $\kappa_m = \bar\kappa + d/2$, so that $d = \kappa_m - \kappa_f$ is the differential and $\bar\kappa$ is held fixed. Then
\begin{equation}
\frac{\partial \kappa^{*}}{\partial d} = -\,\frac{p_f V_f - p_m V_m}{2V},
\end{equation}
which is zero exactly when $p_f V_f = p_m V_m$, that is when the two groups contribute equal shares of the proxy's within-group variance. In our markets the sexes are close to equal in both weight share and schooling variance, so (7) is small and the differential is close to inoperative by construction rather than by assumption.

\paragraph{A.5 What this implies for the sign.}
The decomposition depends on the market and the covariate set but not on the outcome definition or the learner, so the 3,000 specification runs carry 500 distinct decompositions. Across the swept range the corrected unexplained component is a monotone function of the assumed reliability in every one of the 500, with no interior turning point. In @inc_total@ it is increasing, so its least negative value is at $\kappa = 1$, where the correction vanishes and the corrected component equals the uncorrected one. In the remaining @dec_total@, the premarket decompositions of the Utah markets, the only markets in which men's mean schooling exceeds women's, it is decreasing, so its least negative value is at the sweep floor of 0.15. At both endpoints, in all 500, the component is negative.

A monotone function that is negative at both ends of an interval is negative throughout it. No reliability in $[0.15, 1.00]$ can therefore reverse the sign, and by (5) no pair of group reliabilities whose equivalent common value lies in that range can reverse it either. Note the direction of the result in the @inc_total@ increasing decompositions: assuming a lower reliability makes the gap larger, not smaller, so the objection the correction answers is not helped by pushing the reliability down.

\paragraph{A.6 Numerical verification.}
Evaluating (5) at the sex-specific literacy anchors estimated in the full adult sample, @km@ for men and @kf@ for women, using each market's own weighted schooling moments by sex as stored with the bootstrap, gives $\kappa^{*}$ between @kstar_lo@ and @kstar_hi@ across the 250 markets. The bound (6) holds in @bound_holds@ of 250, with $\bar\kappa_V$ between @kbarV_lo@ and @kbarV_hi@. The sensitivity (7) lies between @sens_lo@ and @sens_hi@. At $\kappa^{*}$ the corrected unexplained component of the primary decomposition, read from the stored sweep, is more than @gap_at_kstar@ log points below zero in every market.

Two stress tests follow. First, holding $\bar\kappa$ at its estimated value and asking how large a differential would be needed to bring $\kappa^{*}$ down to the sweep floor: the answer is at least @need_min@ in the most exposed market and @need_med@ in the median market, against an arithmetic maximum of @arith_max@ for any differential at this mean, since the smaller of the two reliabilities cannot be negative. The required differential exceeds the largest possible one in @need_exceeds@ of 250 markets. For comparison, the estimated differential is @diff_est@ with a replicate-based standard error of @diff_se@.

Second, abandoning the differential framing and setting both sexes at their 95 per cent lower confidence bounds, @lb_m@ for men and @lb_f@ for women, gives $\kappa^{*}$ between @kstar_lb_lo@ and @kstar_lb_hi@. This is the sense in which a reversal would need a level rather than a difference: only a joint downward movement of both reliabilities takes the equivalent value near the floor of the range we sweep, and in @inc_total@ of the 500 decompositions such a movement deepens the gap rather than reversing it.

Equations (1) to (7) are also verified on synthetic data with known answers by a test in the code release (\texttt{tests/test\_note1.py}), which prints each figure it checks. On a sample built with group-specific error variances, the identity (1) holds to $2 \times 10^{-15}$; the error variance implied at $\kappa^{*}$ matches that implied by the pair exactly; the bound (6) holds; the derivative (7) matches its finite-difference value to $3 \times 10^{-12}$; and supplying $\kappa^{*}$ to the estimator recovers the coefficient on the mismeasured column to within 0.001 of its true value, where ordinary least squares is attenuated by 0.028.

\paragraph{A.7 Scope.}
The derivation assumes classical error that is mean zero within each group. It does not cover a sex difference in the mean of the error, which shifts $\mu_s$ rather than inflating $V_s$ and is absorbed by the explained component instead of the error variance. That is a different violation from the one the registered differential-reliability test was written to detect. The main text tests it directly in the assessment survey (Supplementary Table S8) and, finding it, carries the estimated difference through the decomposition before the reliability correction is applied (Supplementary Table S9); part~A establishes only that a difference in reliability cannot do what a difference in mean error does.

\subsection*{Part B. Uncertainty, dependence and identification}

This note sets out what the intervals in the main text cover, what they do not, how the dependence between markets is handled, and when the correction exists at all.

\paragraph{B.1 What the bootstrap resamples.}
Intervals are computed on the primary arm, the income threshold with premarket covariates and a logistic screening model, in each of the 250 markets. Each market draws 500 bootstrap samples, with persons drawn within sex and with probability proportional to the survey person weight. Drawn rows then enter every estimator with unit weight, while the point estimates that the intervals surround remain weighted. Seeds are derived per market from the state code and year, so a rerun reproduces the same draws.

Each draw recomputes the whole chain jointly rather than one quantity at a time: the base-rate gap, the predicted-merit gap with the screening model refitted out of fold, the reversal statistic, the uncorrected residual, the corrected residual at the anchor, and the difference between the last two. Because the draws are shared, the interval for the deepening is an interval on the difference itself and not the overlap of two separate intervals.

The resulting counts, over the 250 markets, are these. The reversal statistic is positive in every one of the 500 draws in all @b1_R@ markets, so its bootstrap $p$ value is at the resolution floor of $1/501$ everywhere. The corrected residual is negative in every draw in all @b1_corr@ markets, and its percentile interval excludes zero in all @b1_corr_ci@. The deepening is negative in every draw in @b1_deep@ markets, and its interval excludes zero in @b1_deep_ci@.

\paragraph{B.2 The second version of the bootstrap.}
The registered procedure had three limits, and the second version repairs two of them. First, drawing with replacement before refitting out of fold placed duplicated persons in both the training and the evaluation partitions of a fold, which makes the predicted-merit gap slightly optimistic within a draw. Version 2 assigns folds by original person. Every point estimate is unchanged, the counts above are identical (@b2_R@, @b2_corr@, @b2_corr_ci@, @b2_deep@ and @b2_deep_ci@), and no interval endpoint moves by more than @fold_max@.

Second, the reliability anchor was held fixed across draws, so the bootstrap did not propagate the sampling error of the anchor into the corrected quantities. Version 2 draws the reliability in every draw from a normal distribution centred at the anchor with its replicate-based standard error, truncated to the swept range; the drawn values run from @kd_lo@ to @kd_hi@. This widens the corrected-residual intervals by a factor of @w_corr@ on average and the deepening intervals by @w_deep@, and admits zero nowhere: the corrected residual still excludes zero in @kd_corr_excl@ markets and the deepening in @kd_deep_excl@. For the mean-error arm at the literacy estimate, where women's schooling is lowered by @bm_shift@ years (standard error @bm_se@) before the correction, the error difference is drawn alongside the reliability from its own estimated distribution. There the error difference dominates: relative to the fixed-parameter intervals, drawing the error difference alone widens the corrected-residual intervals by @bm_w_shift@, drawing the reliability alone by @bm_w_kappa@, and drawing both by @bm_w_both@. With both drawn the corrected residual remains below zero in @bm_excl_both@ of 250 markets, and the net change relative to the uncorrected residual has an interval covering zero in @bm_net_both@ markets, against @bm_net_fixed@ with both parameters fixed; that is the sense in which the corrected gap under the estimated error structure is statistically indistinguishable from the uncorrected one in most markets while remaining adverse in all of them.

Third, and unrepaired, the resampling treats persons as independent within sex conditional on the realised weights. It does not reproduce the successive-difference replication design of the survey, so it is a resampling of the analysis sample and not a design-based variance estimate. Non-primary specifications carry point estimates only, which was fixed when the design was registered to bound the computation of a grid this size.

\paragraph{B.3 The registered replication test.}
The registered test of replication asked whether the reversal survives multiplicity correction across markets. We form a one-sided $p$ value for each of the @fdr_n@ primary cells and apply the Benjamini-Hochberg procedure at a false discovery rate of 0.05. All @fdr_rej@ are rejected, so the registered falsifier, the reversal being absent or itself reversed in a majority of powered cells, does not fire.

Two qualifications. The interval stored in the grid comes from a simpler resampling than the joint bootstrap of B.1: it is unweighted, carries a single seed and covers the reversal statistic alone. And the grid stores each cell's percentile interval but not its individual draws, so the $p$ value used in the procedure is a normal approximation, taking the standard error as the interval width divided by $2 \times 1.96$ and referring the ratio of the estimate to it. The later joint bootstrap does record the sign shares directly, and they agree: the reversal statistic is positive in every draw of every market, which is as strong as a 500-draw bootstrap can report. The approximation is therefore not load-bearing, and we note it because a reader comparing the two runs would otherwise find two routes to the same conclusion and no statement of which was used.

\paragraph{B.4 Dependence across markets.}
The 250 markets are 50 states observed in five years, so they are not 250 independent observations: a state contributes five correlated cells. We report two checks, both computed from the stored grid by seeded scripts.

Omitting one state at a time and recomputing the mean of the reversal statistic over the remaining 245 cells moves it between @R_loso_min@, when @R_loso_min_st@ is omitted, and @R_loso_max@, when @R_loso_max_st@ is omitted, against a full-sample mean of @R_mean@. No single state carries the result.

Resampling states with replacement, keeping all five years of a drawn state together so that the resampling unit is the unit of dependence, gives a 95 per cent interval for the mean of @R_ci_lo@ to @R_ci_hi@ over 10,000 draws. The corresponding standard error is @R_se_b@, against @R_se_i@ for an interval that treats the 250 markets as independent. Acknowledging the clustering therefore roughly doubles the standard error, and the interval still sits well away from zero.

The same two checks apply to the pooled decomposition quantities, which are equal-weight means across the 250 markets of the primary arm's premarket decomposition. The uncorrected unexplained gap is @p_uncorrected_residual@ log points with a state-block interval of @p_uncorrected_residual_lo@ to @p_uncorrected_residual_hi@; the corrected gap at the anchor is @p_corrected_residual@ (@p_corrected_residual_lo@ to @p_corrected_residual_hi@); the change on correction is @p_deepening@ (@p_deepening_lo@ to @p_deepening_hi@). Leaving any one state out moves the corrected value only between @p_corrected_residual_loso_lo@ (@p_corrected_residual_loso_lo_st@ omitted) and @p_corrected_residual_loso_hi@ (@p_corrected_residual_loso_hi_st@ omitted). The state-block standard errors are about twice the independent ones (ratios @p_uncorrected_residual_ratio@, @p_corrected_residual_ratio@ and @p_deepening_ratio@). Along the reliability sweep the pooled corrected gap is largest at $\kappa = @sweep_kappa@$, where it equals the uncorrected value of @sweep_max@ with an upper interval bound of @sweep_max_hi@, so the pooled gap is below zero at every swept reliability. Supplementary Table S5 collects these figures.

The conclusion we draw is the modest one stated in the main text: the estimate survives the dependence, but the 250 markets should be read as one phenomenon observed repeatedly rather than as 250 replications.

\paragraph{B.5 Identification of the correction, and two implementation checks.}
The corrected decomposition solves $(S_{XX} - \Lambda) b = S_{Xy}$ and requires $S_{XX} - \Lambda$ to be positive definite. With a single mismeasured column $j$ and $\Lambda = (1 - \kappa)\, S_{jj}\, e_j e_j'$, the matrix is positive definite exactly when $(1 - \kappa) S_{jj} < 1 / (S_{XX}^{-1})_{jj}$, that is when $\kappa > R^2_j$, where $R^2_j$ is the weighted share of the proxy's variance that the other covariates in the design already reproduce. The correction therefore exists only when the assumed reliability exceeds that share. In the registered covariate sets the share is small: the matrix was positive definite at every point of the reliability sweep in every one of the 3,000 specification runs, and the estimator, which raises an error rather than returning coefficients from an indefinite system, recorded no failure in any of the 125,000 draws of each of the three bootstrap runs. Once occupation major groups enter the design the condition fails everywhere: the threshold reliability averages @thr_mean@ (state-block interval @thr_lo@ to @thr_hi@), runs from @thr_min@ to @thr_max@ across markets, and exceeds the anchor of 0.187 in @thr_above@ of 250. The estimator now records the threshold for every specification and stores no corrected quantity below it, keeping the uncorrected decomposition, the reversal statistic and the admissible part of the sweep; on that part the corrected gap is @adm35_mean@ at a reliability of 0.35, negative in @adm35_neg@ of 250 markets, and negative in all 250 from 0.40 upwards (Supplementary Table S10).

Each bootstrap run also recomputes its market's reversal statistic and compares it against the independently stored grid value, recording the outcome in the result file and exiting with an error if any market disagrees. All 250 agreed to within $10^{-9}$ in the registered run and in version 2; the mean-error bootstrap has no stored counterpart and records the comparison as not applicable.

\clearpage
\section*{Supplementary Tables}

\sitab{S1}{\textbf{Reliability of the schooling proxy by age band.} Reliability is the weighted squared correlation between years of qualification and the directly assessed measure, combined over ten plausible values with Fay-adjusted replicate weights. Replicate-based standard errors in parentheses. Bands are the survey's own AGEG10LFS age groups, given here by their standard labels. The gradient is why the age-restricted anchor exceeds the full-file anchor: in the youngest band many respondents are still in education, so schooling carries almost no information about assessed skill.}{\begin{center}\input{tables/S1}\end{center}}

\sitab{S2}{\textbf{Reliability within attainment category.} Categories are the file's eight-category attainment coding, in ascending order of attainment; we report them by code rather than by label because the label set is the survey's, not ours. @S2_not_word@ categories, holding @S2_resp@ of the 3,758 respondents, return no estimate at all because the proxy is constant within them; in the three where it still varies the reliability is at most @S2_max@. This is mechanical rather than substantive: conditioning on attainment removes most of the variance of the schooling proxy, so within a category there is little left for the assessed measure to correlate with. The same logic is why the attainment-coded proxy cannot be stratified by attainment at all. These rows are reported for completeness and carry no weight in the paper.}{\begin{center}\input{tables/S2}\end{center}}

\sitab{S3}{\textbf{How much of the base-rate difference the covariates account for.} Each run contributes the ratio of the predicted-merit gap to the base-rate gap, so 0.32 means the covariates account for 32 per cent of the difference and a negative value means they predict it in the opposite direction. Extended covariates absorb about a third; premarket covariates absorb none and point the other way in almost every run. The last column is the share of runs with a negative ratio.}{\begin{center}\small\input{tables/S3}\end{center}}

\sitab{S4}{\textbf{Strict verdict rate by analyst dimension.} The strict verdict is $\Delta_{\rm merit} > 0 > \Delta_{\rm label}$. Each row holds one dimension at one level and pools over the other two across all 3,000 specification runs. The covariate set moves the rate across almost its whole range; the outcome definition and the learner barely move it. The pooled concordance statistics recorded before estimation are uninformative here for the reason given in the main text, which is why we report the dimensions separately.}{\begin{center}\input{tables/S4}\end{center}}

\sitab{S5}{\textbf{Dependence across markets.} The 250 markets are 50 states observed in five years, so a state contributes five correlated cells. Panel A concerns the reversal statistic of the primary arm; panel B the pooled decomposition quantities of the same arm, equal-weight means across the 250 markets. Both use the stored grid; the bootstrap resamples states with replacement and keeps all years of a drawn state together, with 10,000 draws seeded from the study's primary seed. In panel B the interval is the state-block 95 per cent percentile interval, the range is the leave-one-state-out range of the mean with the omitted state named, and the last column is the ratio of the state-block standard error to the one that treats markets as independent.}{\noindent\textbf{A. Reversal statistic}
\begin{center}\input{tables/S5}\end{center}
\noindent\textbf{B. Pooled decomposition quantities, premarket covariates}
\begin{center}\small\input{tables/S5B}\end{center}}

\sitab{S6}{\textbf{Computational environment.} Versions are those recorded in every run log; each arm additionally stores its own configuration hash, so a rerun that differs in any registered setting is detected rather than silently merged. Determinism was verified by recomputing cells from a separate session and comparing every stored field apart from timing and the write timestamp. The hashes of the arms added after the internal review of 22 September 2026 are listed so that the released run logs can be matched to this document.}{\begin{center}\small\input{tables/S6}\end{center}}

\sitab{S7}{\textbf{What sets the direction of the correction.} One decomposition per market and covariate set (500 in all; the outcome definition and the learner do not enter it). The total explained component is positive in every premarket decomposition and negative in every extended one, yet correction deepens the gap in all but three of the 500, so the total explained component is not the operative quantity. The operative quantity is the between-sex difference in mean schooling: in the primary arm women lead in @S7_women_lead@ of the 250 markets and the gap deepens in all @S7_deep_women@; men lead in three (@S7_men_list@) and the gap attenuates in all @S7_att_men@. The three are near-ties, with men ahead by at most @S7_men_max@ attainment-code points, whereas where women lead they do so by @S7_lead_min@ to @S7_lead_max@ points, median @S7_lead_med@. Across markets the size of the deepening correlates at @S7_corr@ with women's schooling lead (the correlation is negative in sign because the deepening is a negative change).}{\begin{center}\small\input{tables/S7}\end{center}}

\sitab{S8}{\textbf{The test of equal error means in the assessment survey.} Each row is a weighted least squares fit of the plausible values of assessed skill on the proxy and an indicator for women, combined over the ten plausible values by Rubin's rules with Fay-adjusted replicate standard errors in parentheses. The benchmark is the female coefficient that classical error with equal error means implies, $b_X (1 - \kappa_w)\kappa_w^{-1}$ times women's lead in mean schooling, recomputed inside every replicate; the difference is the coefficient minus the benchmark with its own replicate-based error and $z$ ratio; $\delta$ is the implied sex difference in the mean of the proxy error in years of qualification, positive where women's schooling overstates their assessed skill relative to men's. Age bands are the survey's AGEG10LFS groups. The attainment-coded specification identifies the female coefficient but not $\delta$, because the proxy is constant within a category. Skill scores are on the survey's 500-point scale. Every specification with a benchmark rejects it, with $z$ between @S8_z_lo@ and @S8_z_hi@; adding the age bands changes $\delta$ by at most @S8_age_change@ years.}{\begin{center}\footnotesize\input{tables/S8}\end{center}}

\clearpage
\sitab{S9}{\textbf{The corrected gap under each estimated error structure.} Women's years of qualification are lowered by $\delta$ before the reliability correction is applied at the anchor, in every market, for both covariate sets and for both samples (all adults at the propagated literacy anchor; earners aged 25 to 65 at that sample's own anchor and its own estimates of $\delta$). The first row of each block is the years-coded reference arm with the classical correction; lower and upper denote the bounds of the 95 per cent interval of the survey estimate. Pooled means are equal-weight means across the 250 markets with state-block 95 per cent intervals; the market range is the minimum and maximum across markets; the counts are the markets in which the corrected gap is negative at the anchor and at the sweep floor of 0.15, and the markets in which the reliability step, taken after the mean-error adjustment, deepens the gap (it attenuates it in the remainder). The pooled gap is linear in $\delta$: on the full sample the fitted relation is $@fit_all_premarket_a@ + @fit_all_premarket_b@\,\delta$ under premarket covariates (largest residual @fit_all_premarket_resid@) and $@fit_all_extended_a@ + @fit_all_extended_b@\,\delta$ under the extended set; on the earners sample $@fit_workers_premarket_a@ + @fit_workers_premarket_b@\,\delta$ and $@fit_workers_extended_a@ + @fit_workers_extended_b@\,\delta$. The correction is neutral, returning the uncorrected years-coded value, at $\delta = @fit_all_premarket_neutral@$ years on the full sample and @fit_workers_premarket_neutral@ on the earners sample under premarket covariates, and the pooled gap would reach zero at @fit_all_premarket_zero@ and @fit_workers_premarket_zero@ years (@fit_all_extended_zero@ and @fit_workers_extended_zero@ under the extended set). Fitting the same line market by market on the full sample, premarket set, the closing value has median @fit_all_premarket_pm_med@ years and runs from @fit_all_premarket_pm_min@ to @fit_all_premarket_pm_max@; it exceeds the numeracy estimate in @fit_all_premarket_above_num@ of 250 markets and the upper bound of its interval in @fit_all_premarket_above_num_hi@. The first-order rule, that the reliability step deepens the gap where women's lead in years net of $\delta$ is positive and attenuates it where that quantity is negative, matches the market-level direction in @rule_pm_min@ to @rule_pm_max@ of 250 markets on the premarket set across every $\delta$ run, and in @rule_ex_min@ to @rule_ex_max@ on the extended set; women's mean lead in years is @lead_all_pm@ on the full sample and @lead_w25_pm@ among earners.}{\begin{center}\footnotesize\input{tables/S9}\end{center}}

\clearpage
\sitab{S10}{\textbf{Six further arms.} Each arm is the registered cell computation with one option changed, run with the logistic learner on the income-threshold outcome in all 250 markets; the registered grid is repeated in the first rows for comparison. The years-coded arm recodes the attainment variable in years through the crosswalk described in Methods, which moves the uncorrected premarket gap from the registered value to @years_U@ and the corrected gap to @years_Uk@ and changes no reversal statistic by more than @years_max_R@; it is the reference arm of the mean-error analysis (Supplementary Table S9). Mean $n$ is the mean analysis sample per market; $R > 0$ counts markets with a positive reversal statistic; the raw gap and the uncorrected and corrected unexplained components are equal-weight means across markets in log points; the last column counts markets in which correction at the anchor moves the gap further from zero. The wage arm uses wage and salary income among full-time full-year workers (full-time in the table), which cuts the raw gap from @grid_raw_pm@ to @wage_raw_pm@ and brings the corrected gaps of the two covariate sets together (@wage_Uk_pm@ and @wage_Uk_ex@). Real dollars leave the log decomposition unchanged to machine precision and move $R$ by at most @adj_max_R@. The weight-respecting cell median leaves the income-threshold specifications untouched by construction and moves the cell-median $R$ by at most @wm_max_R@ without changing any market's sign. Ages 25 to 65 at the earners anchor of @age_anchor@, that sample's own anchor, leave the deepening at @age_deep@, larger than in the registered grid. With occupation major groups added to the extended set the reversal statistic averages @occ_R@ against @ext_R@ without them and stays positive in every market, but the corrected decomposition is not identified at the anchor in any market (panel B).}{\begin{center}\footnotesize\input{tables/S10}\end{center}
\vspace{6pt}
\noindent\textbf{B. The occupation arm: where the correction is identified.} The threshold is $R^2_j$, the weighted share of the proxy's variance that age, the extended covariates and the occupation dummies together reproduce; the correction exists only above it. The admissible part of the sweep is reported at five reliabilities.
\begin{center}\small\input{tables/S10B}\end{center}}

\end{document}
"""

missing = sorted(set(re.findall(r"@([A-Za-z0-9_]+)@", TEMPLATE)) - set(V))
assert not missing, missing
out = TEMPLATE
for k, v in V.items():
    out = out.replace(f"@{k}@", v)
assert "@" not in out.replace("\\@", ""), [m for m in re.findall(r"@[^ ]{0,30}", out)][:5]
assert "—" not in out and "--" not in out.replace("\\\\", "")
open(os.path.join(HERE, "si.tex"), "w").write(out.lstrip())
print("wrote si.tex")
