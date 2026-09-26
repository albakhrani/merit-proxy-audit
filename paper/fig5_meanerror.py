#!/usr/bin/env python3
"""Figure 5: the pooled corrected gap against the sex difference in mean proxy
error, on the full sample and on the earners sample. Every number is read
from the run logs in results/; nothing is typed in.

Usage, from the repository root: python paper/fig5_meanerror.py [results_dir] [out_stem]
(results_dir defaults to results/ next to paper/, out_stem to paper/fig5_meanerror;
the figure is written as out_stem.pdf, .eps and .png)."""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5_meanerror")

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
                     "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.6, "xtick.color": "#444444", "ytick.color": "#444444",
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def cells(d):
    """{(state, year, covset): corrected unexplained at the anchor} for acs50k."""
    out = {}
    for f in sorted(glob.glob(os.path.join(R, d, "*.json"))):
        c = json.load(open(f))
        for s in c["spec_results"]:
            if s["outcome_def"] == "acs50k":
                out.setdefault((c["state"], c["year"], s["covariate_set"]), s["eiv_unexplained_at_kappa"])
    return out


def pooled(summary, tag, cs, key):
    p = summary["arms"][tag]["covsets"][cs]["pooled"][key]
    return p["mean"], p["ci_lo"], p["ci_hi"]


ms = json.load(open(os.path.join(R, "meanerror_summary.json")))
mw = json.load(open(os.path.join(R, "meanerror_summary_workers.json")))
pv = json.load(open(os.path.join(R, "piaac_validity.json")))["specs"]

def piaac(skill, sample):
    d = pv[skill][sample]["years_noage"]["delta_proxy_units"]
    return d["point"], d["se"]

panels = [
    dict(title="All adults, reliability 0.19", years="grid_years",
         arms=[("lit_lo", "grid_meanerror_lit_lo"), ("lit", "grid_meanerror_lit"), ("lit_hi", "grid_meanerror_lit_hi"),
               ("num", "grid_meanerror_num"), ("num_hi", "grid_meanerror_num_hi")],
         summary=ms, base_tag="lit", piaac=[("literacy", *piaac("lit", "all")), ("numeracy", *piaac("num", "all"))],
         xmax=0.95),
    dict(title="Earners aged 25 to 65, reliability 0.24", years="grid_years_age2565",
         arms=[("lit_workers", "grid_meanerror_lit_workers"), ("num_workers", "grid_meanerror_num_workers")],
         summary=mw, base_tag="lit_workers", piaac=[("literacy", *piaac("lit", "workers")), ("numeracy", *piaac("num", "workers"))],
         xmax=1.32),
]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.14, wspace=0.08)
record = {}
for ax, P, letter in zip(axes, panels, "ab"):
    S = P["summary"]
    base = cells(P["years"])
    arms = [(S["arms"][t]["delta_years"], t, cells(d)) for t, d in P["arms"]]
    # PIAAC estimate bands first, behind everything
    for name, pt, se in P["piaac"]:
        ax.add_patch(Rectangle((pt - 1.96 * se, -1.32), 2 * 1.96 * se, 1.80, color="#efefec", zorder=0, lw=0))
        ax.plot([pt, pt], [-1.32, 0.40], color="#9a9a96", lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(pt, 0.415, name, ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.axhline(0, color=INK, lw=0.9, zorder=2)
    ax.grid(True, axis="y", zorder=0)
    for cs, colour, label in (("premarket", BLUE, "premarket covariates"), ("extended", ORANGE, "extended covariates")):
        xs = [0.0] + [d for d, _, _ in arms]
        means, los, his = [], [], []
        m, lo, hi = pooled(S, P["base_tag"], cs, "U_yk"); means.append(m); los.append(lo); his.append(hi)
        for d, t, _ in arms:
            m, lo, hi = pooled(S, t, cs, "U_mk"); means.append(m); los.append(lo); his.append(hi)
        # range across the 250 markets
        vmin = [min(v for (s, y, c), v in base.items() if c == cs)] + [min(v for (s, y, c), v in a.items() if c == cs) for _, _, a in arms]
        vmax = [max(v for (s, y, c), v in base.items() if c == cs)] + [max(v for (s, y, c), v in a.items() if c == cs) for _, _, a in arms]
        ax.fill_between(xs, vmin, vmax, color=colour, alpha=0.10, lw=0, zorder=2)
        # linear fit through the pooled points and its zero crossing
        b, a0 = np.polyfit(xs, means, 1)
        zero = -a0 / b
        if zero > max(xs):
            ax.plot([max(xs), zero], [a0 + b * max(xs), 0], color=colour, lw=1.0, ls=(0, (1.5, 2)), zorder=3)
        ax.plot(xs, means, color=colour, lw=1.8, zorder=4, label=label)
        ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    fmt="o", ms=4, color=colour, ecolor=colour, elinewidth=1.0, capsize=2, zorder=5)
        ax.plot([zero], [0], marker="v", ms=5, color=colour, zorder=6, clip_on=False)
        ax.text(zero - 0.014, 0.035, f"{zero:.3f} yr", ha="right", va="bottom", fontsize=7.5, color=colour,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=7)
        neg_anchor = {d: sum(1 for (s, y, c), v in a.items() if c == cs and v < 0) for d, _, a in arms}
        record[f"{letter}_{cs}"] = {"deltas": xs, "pooled": means, "ci_lo": los, "ci_hi": his, "min": vmin, "max": vmax,
                                     "fit_intercept": a0, "fit_slope": b, "zero_crossing": zero, "negative_at_anchor": neg_anchor}
    ax.set_xlim(-0.02, P["xmax"]); ax.set_ylim(-1.32, 0.48)
    ax.set_xlabel("Sex difference in mean proxy error (years)")
    ax.set_title(P["title"], loc="left", fontsize=9, pad=14)
    ax.text(-0.14 if letter == "a" else -0.05, 1.10, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
axes[0].set_ylabel("Corrected unexplained gap (log points)")
axes[0].legend(loc="lower right", frameon=False, handlelength=1.6)
for ext in ("pdf", "eps", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300 if ext == "png" else None)
json.dump(record, open(f"{OUT}_data.json", "w"), indent=1)
print("wrote", OUT)
for k, v in record.items():
    print(f"  {k}: deltas {[round(x,3) for x in v['deltas']]} pooled {[round(x,3) for x in v['pooled']]} zero {v['zero_crossing']:.3f} negative-at-anchor {v['negative_at_anchor']}")
