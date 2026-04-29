# PHM Revision Submission Package

This directory contains the comprehensively revised manuscript for submission to **Population Health Management** (Sage). The work was previously evaluated at PLOS Medicine; the present version was rebuilt to address all four reviewers' concerns.

## Title

*Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study*

## What changed from the prior submission

| Reviewer concern | Resolution in this version |
|---|---|
| "Doubly robust" estimator was actually plug-in plus a constant | Replaced with **causal forest** (primary), **DR-Learner** (Kennedy 2023), **R-Learner**, with cross-fitting per Chernozhukov et al. T- and S-learners reported as ML benchmarks |
| Time-varying confounding not addressed | Added **MSM with stabilized IPTW** and **target-trial emulation** |
| Variance decomposition decomposed *outcome*, not *treatment effect* | Re-fit mixed model on **estimated CATEs**; reported within-person fraction = 63.6% |
| IV assumptions not interrogated | Added **pretrend test**, **exclusion-restriction probe**, **monotonicity check**, plus **Callaway-Sant'Anna** triangulation |
| "World model" terminology misleading | Removed entirely; analytic schema redrawn |
| Date inconsistency | Single window: Jan 1, 2023 — Dec 31, 2025 (36 months) |
| Figure 1 had undefined `z_i` and "distilled CATE" | All notation defined inline |
| Post-hoc power analysis | Removed (per Althouse 2020); replaced with CI-precision reporting |
| References broken (gaps, mismatches) | All 45 references verified; numbered in order of appearance |
| Code not public | All code in this repo, public, with synthetic-data demo |

## Files

- `manuscript.md` / `manuscript.docx` — main text with embedded figures
- `appendix.md` / `appendix.docx` — supplementary appendix (S1-S4)
- `cover_letter.md` / `cover_letter.docx` — cover letter
- `figures/Figure_1_Schema.png` — analytic schema (replaces prior z_i / distilled-CATE diagram)
- `figures/Figure_2_CATE_and_Contrast.png` — CATE distribution + primary contrast
- `figures/Figure_3_RATE_and_IV.png` — RATE curve + IV-eligible tertile validation
- `make_figures.py` — figure-generation script (300 DPI)
- `build_docx.sh` — pandoc build script

## Reproducibility

A synthetic-data demonstration that re-runs every figure and table on simulated data is in the repository root (`code/` and `outputs/`). Individual-level Medicaid data are not shareable per data-use agreements with state agencies.
