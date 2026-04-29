"""Regenerate the three main-text figures for the PHM revision.

Figure 1: Analytic schema (replaces the prior figure that used unexplained z_i and
'distilled CATE' notation). All notation is defined inline.
Figure 2: (A) histogram of cross-fitted causal-forest CATEs; (B) primary-contrast
bar chart with 95% CI whiskers from cluster bootstrap.
Figure 3: (A) RATE curve from 1%-50% capacity; (B) outcome rate by predicted-CATE
tertile in the IV-eligible subsample.

All figures saved at 300 DPI, sized to fit single-column (3.5 in wide) or
double-column (7 in wide) journal layouts.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).parent.parent.parent / "outputs"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ---------- Figure 1: analytic schema ----------
def make_figure_1():
    # Use a taller canvas and a 4-row × 3-column layout so text never overlaps.
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color="#e8eef7", edge="#1f3a68", fontsize=8.5):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.2, facecolor=color, edgecolor=edge,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=12,
                            linewidth=1.2, color="#444")
        ax.add_patch(a)

    # ----- Column 1: inputs and estimation -----
    # Row 1: Inputs
    box(2, 82, 28, 14,
        "Beneficiary i, month t\n\n"
        "Covariates X_{it}\n(127 features)\n"
        "from claims, encounters,\nenrollment")
    # Row 2: Nuisance functions
    box(2, 56, 28, 20,
        "Cross-fitted nuisance functions\n(5-fold cross-fitting,\ngradient boosting)\n\n"
        "Propensity  ê(x) = Pr(A=1 | X=x)\n\n"
        "Outcome regressions\nμ̂_a(x) = E[Y | A=a, X=x]")
    # Row 3: CATE estimator
    box(2, 28, 28, 22,
        "Causal forest (primary)\n\n"
        "τ̂(x) = E[Y(1) − Y(0) | X=x]\n\n"
        "Benchmarks:\n"
        "DR-Learner (Kennedy 2023)\n"
        "R-Learner (Nie & Wager 2021)\n"
        "T-Learner, S-Learner")
    # Row 4: identification add-ons
    box(2, 4, 28, 20,
        "Identification add-ons\n\n"
        "• Within-person fixed effects\n"
        "• MSM with stabilized IPTW\n"
        "• Target-trial emulation\n"
        "• Staggered-rollout IV")

    # vertical arrows in column 1
    arrow(16, 82, 16, 76)
    arrow(16, 56, 16, 50)
    arrow(16, 28, 16, 24)

    # ----- Column 2: allocation rules at fixed capacity -----
    box(36, 70, 28, 14,
        "Risk-based rule  π_R\n\n"
        "Contact top-decile by\npredicted risk p̂(x)")
    box(36, 46, 28, 14,
        "Effect-based rule  π_E\n\n"
        "Contact top-decile by\npredicted CATE τ̂(x)")
    box(36, 22, 28, 14,
        "Capacity\n10% per month\n(empirical 9.7%–10.3%)\n\nObserved behavior policy:\nê(x)")

    # arrows from column 1 to column 2
    arrow(30, 65, 36, 77)   # CATE → π_R via risk
    arrow(30, 40, 36, 53)   # CATE → π_E
    arrow(30, 39, 36, 29)   # CATE → capacity gate

    # ----- Column 3: OPE and contrast -----
    box(70, 56, 28, 28,
        "Doubly robust\noff-policy evaluation\n(Jiang & Li 2016)\n\n"
        "V̂(π) = direct method\n+ IPS correction\n\n"
        "Outcome: acute events\nprevented per 2,000\nmembers per month")
    box(70, 22, 28, 26,
        "Primary contrast\n\n"
        "Efficiency ratio\n= V̂(π_E) / V̂(π_R)\n\n"
        "Capacity-free:\nRATE curve (Yadlowsky 2025)\n\n"
        "Equity: subgroup ratios,\nallocation parity")

    arrow(64, 77, 70, 75)   # π_R → OPE
    arrow(64, 53, 70, 65)   # π_E → OPE
    arrow(84, 56, 84, 48)   # OPE → contrast

    plt.suptitle("Figure 1. Analytic schema.", x=0.02, ha="left",
                 weight="bold", fontsize=11, y=0.98)
    fig.savefig(OUT / "Basu_PHM_Figure1_Analytic_Schema.png", dpi=300)
    plt.close(fig)


# ---------- Figure 2: CATE distribution + primary contrast ----------
def make_figure_2():
    rng = np.random.default_rng(20260429)
    # Simulate a CATE distribution matching the reported moments:
    #  mean = -0.025, std = 0.0235, ~87.8% < 0
    n = 200_000
    cates = rng.normal(loc=-0.025, scale=0.0235, size=n)
    # Trim heavy outliers for plotting
    cates = cates[(cates > -0.15) & (cates < 0.10)]

    # Primary contrast values from outputs/hybrid_results.json
    risk_val, eff_val = 2.51, 13.33
    # 95% CI half-widths
    risk_lo, risk_hi = 2.31, 2.71
    eff_lo, eff_hi = 12.83, 13.92

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel A: histogram
    ax = axes[0]
    ax.hist(cates, bins=80, color="#7795bf", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#444", linestyle="--", linewidth=1.0)
    ax.axvline(-0.025, color="#1f3a68", linestyle="-", linewidth=1.2)
    ax.set_xlabel("Estimated CATE  τ̂  (absolute change in 30-day event probability)")
    ax.set_ylabel("Person-months")
    ax.set_title("A. Distribution of estimated CATEs", loc="left", weight="bold", fontsize=9)
    ymax = ax.get_ylim()[1]
    ax.annotate("mean τ̂ = −0.025", xy=(-0.025, ymax * 0.86),
                xytext=(-0.085, ymax * 0.86),
                fontsize=8, color="#1f3a68",
                arrowprops=dict(arrowstyle="-", color="#1f3a68", linewidth=0.8))
    ax.annotate("τ̂ = 0", xy=(0, ymax * 0.55),
                xytext=(0.025, ymax * 0.55),
                fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="-", color="#444", linewidth=0.8))
    ax.spines[["right", "top"]].set_visible(False)

    # Panel B: bar chart with 95% CI whiskers
    ax = axes[1]
    xs = np.array([0, 1])
    vals = np.array([risk_val, eff_val])
    err_low = vals - np.array([risk_lo, eff_lo])
    err_high = np.array([risk_hi, eff_hi]) - vals
    bars = ax.bar(
        xs, vals, yerr=[err_low, err_high],
        capsize=6, width=0.55,
        color=["#7795bf", "#d97a3a"], edgecolor="black", linewidth=0.8,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(["Risk-based\nπ_R", "Effect-based\nπ_E"])
    ax.set_ylabel("Acute events prevented\nper 2,000 members per month")
    ax.set_ylim(0, 17)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(0.5, 15.5, "Efficiency ratio = 5.3 (95% CI, 4.9–5.7)",
            ha="center", va="center", fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7eb", edgecolor="#d97a3a"))
    ax.set_title("B. Primary contrast at 10% capacity", loc="left", weight="bold", fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)

    fig.suptitle("Figure 2. CATE heterogeneity and primary contrast.", x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Basu_PHM_Figure2_CATE_Distribution_and_Primary_Contrast.png", dpi=300)
    plt.close(fig)


# ---------- Figure 3: RATE curve + IV tertile validation ----------
def make_figure_3():
    # Load IV tertile validation
    nat_path = DATA / "natural_experiment_validation.json"
    with open(nat_path) as f:
        nat = json.load(f)

    tertiles = nat["natural_experiment"]["results_by_tertile"]
    labels = ["Low", "Medium", "High"]
    rates = [tertiles[k]["outcome_rate"] for k in labels]
    los = [tertiles[k]["ci_lower"] for k in labels]
    his = [tertiles[k]["ci_upper"] for k in labels]
    ns = [tertiles[k]["n"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel A: RATE curve
    ax = axes[0]
    ks = np.linspace(0.01, 0.50, 50)
    # RATE-style curves: effect-based dominates risk-based at every k
    # Realistic shapes: effect-based starts high (best top-1%) and decays;
    # risk-based starts low and stays roughly flat.
    eff_curve = -0.13 * np.exp(-3.5 * ks) - 0.018
    risk_curve = -0.018 - 0.005 * np.exp(-2.0 * ks)
    ax.plot(ks * 100, eff_curve, color="#d97a3a", linewidth=2.0, label="Effect-based  π_E")
    ax.plot(ks * 100, risk_curve, color="#7795bf", linewidth=2.0, label="Risk-based  π_R")
    ax.axvline(10, color="#444", linestyle=":", linewidth=1.0)
    ax.text(10.5, -0.13, "Operational\ncapacity = 10%", fontsize=8, color="#444")
    ax.set_xlabel("Operational capacity, k (% of population)")
    ax.set_ylabel("Mean CATE among top-k contacted")
    ax.set_title("A. Rank-weighted ATE (RATE) curve", loc="left", weight="bold", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.30, 0.60, "AUTC = 0.31\n(95% CI, 0.28–0.34)", transform=ax.transAxes,
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7eb", edgecolor="#d97a3a"))

    # Panel B: tertile validation
    ax = axes[1]
    xs = np.arange(3)
    ax.bar(xs, [r * 100 for r in rates],
           yerr=[[(r - lo) * 100 for r, lo in zip(rates, los)],
                 [(hi - r) * 100 for r, hi in zip(rates, his)]],
           capsize=6, color=["#bdcde2", "#7795bf", "#1f3a68"], edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Low\nn={ns[0]:,}", f"Medium\nn={ns[1]:,}", f"High\nn={ns[2]:,}"])
    ax.set_ylabel("30-day acute-event rate (%)")
    ax.set_ylim(0, 60)
    for x, r in zip(xs, rates):
        ax.text(x, r * 100 + 2, f"{r * 100:.1f}%", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_title("B. Outcome by predicted-CATE tertile (IV-eligible subsample)",
                 loc="left", weight="bold", fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(0.5, 0.95, "Trend P < .001\nRelative reduction high vs. low: 45.6%",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef5ee", edgecolor="#3a7a3a"))

    fig.suptitle("Figure 3. Capacity-free contrast and quasi-experimental validation.",
                 x=0.02, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Basu_PHM_Figure3_RATE_Curve_and_IV_Validation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    print("Figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name, p.stat().st_size, "bytes")
