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
| References broken (gaps, mismatches) | All 40 references verified; numbered in order of appearance; every text citation appears in the bibliography |
| Code not public | All code in this repo, public, with synthetic-data demo |

## Files

Naming convention: `Basu_PHM_<DocumentType>_EffectBased_Targeting_Medicaid.<ext>`. Every file in this directory carries a self-describing name so editors and reviewers can identify it without context.

**Submission artifacts (upload these to the journal portal):**

- `Basu_PHM_Manuscript_EffectBased_Targeting_Medicaid.docx` — main text with embedded figures and figure legends; equations as native Microsoft Word OMML
- `Basu_PHM_Supplement_EffectBased_Targeting_Medicaid.docx` — supplementary appendix (S1 methods, S2 reporting checklists, S3 supplementary tables, S4 supplementary figures)
- `Basu_PHM_CoverLetter_EffectBased_Targeting_Medicaid.docx` — cover letter (acknowledges PLOS provenance, summarizes revision, names six suggested reviewers)
- `figures/Basu_PHM_Figure1_Analytic_Schema.png` — analytic schema (300 DPI; all notation defined inline)
- `figures/Basu_PHM_Figure2_CATE_Distribution_and_Primary_Contrast.png` — histogram of cross-fitted CATEs and primary contrast (300 DPI)
- `figures/Basu_PHM_Figure3_RATE_Curve_and_IV_Validation.png` — rank-weighted ATE curve and quasi-experimental tertile validation (300 DPI)

**Editable sources (`.md` mirror of each `.docx`):**

- `Basu_PHM_Manuscript_EffectBased_Targeting_Medicaid.md`
- `Basu_PHM_Supplement_EffectBased_Targeting_Medicaid.md`
- `Basu_PHM_CoverLetter_EffectBased_Targeting_Medicaid.md`

**Build scripts:**

- `generate_submission_figures.py` — regenerates all three figures from `outputs/`
- `build_submission_docx.sh` — converts `.md` → `.docx` (with embedded figures for the manuscript) using pandoc; preserves OMML equations

## Reproducibility

A synthetic-data demonstration that re-runs every figure and table on simulated data is in the repository root (`code/` and `outputs/`). Individual-level Medicaid data are not shareable per data-use agreements with state agencies.
