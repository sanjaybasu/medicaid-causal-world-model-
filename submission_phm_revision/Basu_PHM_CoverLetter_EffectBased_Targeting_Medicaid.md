# Cover Letter

**Date:** April 29, 2026

David B. Nash, MD, MBA  
Editor-in-Chief, *Population Health Management*  
Sage Publishing

**Re:** New manuscript submission — *Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study*

---

Dear Dr. Nash,

We are pleased to submit our manuscript for consideration as an Original Research article in *Population Health Management*. The study compares two approaches to allocating scarce care-management capacity in two large state Medicaid programs (Washington and Virginia; n = 164,063 adult beneficiaries; 2,670,806 person-months; January 2023 — December 2025) and asks: **does targeting outreach by predicted treatment effect outperform targeting by predicted risk, and is any benefit equitably distributed?**

Using a causal forest with cross-fitted nuisance functions, doubly robust off-policy evaluation, marginal structural model targeting for time-varying confounding, and a staggered-rollout instrumental-variable check, we find that effect-based monthly allocation prevented **5.3 (95% CI, 4.9 to 5.7) times more acute events** at the same 10% program capacity. Sixty-four percent of variance in *estimated treatment effects* was within-person across months — that is, much of what determines whether outreach helps a given beneficiary lies in the *moment* rather than the *person*. Efficiency gains were similar across racial/ethnic groups, and allocation parity was preserved.

### Fit with *Population Health Management*

Two recent issues of the journal — Pourat et al. on California Medicaid managed care (2025;28(3):117-124) and Sakinah et al. on virtual urgent care for Medicaid and dual-eligible beneficiaries (2025;28(3):173-178) — show the journal's continuing engagement with retrospective evaluations of how Medicaid programs deploy clinical and operational interventions to high-need populations. Our study fits this scope directly: the question is operationally consequential for every Medicaid managed-care plan and ACO that runs care management on a fixed labor budget, and the methods are now mature enough that adoption is plausible without academic infrastructure.

### Prior consideration and substantive revision

This work was previously evaluated at *PLOS Medicine* (PMEDICINE-D-26-00411R1) and was not accepted in its earlier form. Four reviewers raised detailed methodological concerns. We have rebuilt the analysis from the ground up to address each concern, and we are sufficiently confident in the revisions that we believe the prior reviewers — particularly the methods reviewers — would now find the work substantially strengthened. Specifically:

- **CATE estimator.** The earlier "doubly robust" estimator was, as one reviewer correctly noted, a plug-in T-learner plus a sample-mean correction, which is neither doubly robust nor a consistent CATE estimator. The primary estimator is now a **causal forest with honest splitting** (Athey-Tibshirani-Wager), benchmarked against the **DR-Learner of Kennedy (2023)** and the **R-Learner of Nie & Wager**, with cross-fitting per Chernozhukov et al. (2018). T- and S-learner estimates are reported as ML benchmarks. All differences are now properly documented.
- **Time-varying confounding.** Within-person fixed effects address time-invariant confounding only. We now also report results under **stabilized IPTW with marginal structural models** (Robins-Hernán-Brumback) and a **target-trial emulation** (Hernán-Robins). The efficiency ratio is stable across all three identification strategies (5.1×–5.3×).
- **Variance decomposition.** The earlier formulation decomposed *outcome* variance, not *treatment-effect* variance — an error one reviewer flagged. The revised analysis fits a person-level mixed model **on the estimated CATEs** themselves and reports the within-person fraction (63.6%, 95% CI, 61.9 to 65.4) directly.
- **Quasi-experimental validation.** The IV analysis now reports an explicit **pretrend test** (slope difference $-0.0004$/month; 95% CI includes zero), an **exclusion-restriction probe** for co-occurring policy/coverage changes ($P = .27$), a **monotonicity assessment**, and a triangulating **Callaway-Sant'Anna staggered DiD** estimate that is consistent with the IV LATE.
- **Framing and terminology.** The "world model" terminology — which two reviewers found misleading and one correctly noted does not match the current ML literature — is **removed** entirely. The auxiliary trajectory simulator is replaced by MSM/IPTW and target-trial emulation, which provide stronger handling of time-varying confounding without requiring a generative simulator. The title and framing now describe the work plainly: a causal-ML targeting study.
- **Specifics flagged in review.** Study period is now consistent everywhere (Jan 1, 2023 — Dec 31, 2025; 36 months); n = 2,670,806 person-months as in the data; Figure 1 is redrawn with all notation defined inline; the analytic time-step is calendar month and stated upfront; the "synthetic demographics" note has been corrected (race/ethnicity is self-reported throughout); descriptives now include intervention frequency per beneficiary; the post-hoc power statement is removed and replaced with CI-precision reporting per Althouse (2020); restricted-outcome sensitivity using the NYU non-deferrable-ED algorithm is added. All references are renumbered, and every text citation now appears in the bibliography.
- **Code availability.** All analysis code, container specifications, hyperparameter grids, and a synthetic-data demonstration are publicly available at `https://github.com/sanjaybasu/medicaid-causal-world-model-` (archived with DOI on acceptance).

### Suggested reviewers

We suggest the following early-career reviewers, none of whom were approached for this work at any prior journal. Each is at the assistant- or recently-promoted-associate-professor stage, and each works directly on the methods (causal machine learning, doubly robust estimation, longitudinal CATEs) or on the application area (Medicaid targeting, ML for healthcare risk adjustment) that the manuscript depends on. We expect they will engage substantively and constructively with the work.

1. **Maggie Makar, PhD** — University of Michigan, Assistant Professor of Electrical Engineering and Computer Science; NSF CAREER awardee for causally motivated machine learning; works directly on "evaluating the effectiveness of population- and individual-level interventions while addressing data disparities" — a near-exact restatement of our research question. Email: mmakar@umich.edu
2. **Anna Zink, PhD** — Tufts University, Assistant Professor of Community Health (since 2024; previously Principal Researcher at the Chicago Booth Center for Applied AI). Built a Medicaid risk-tiering algorithm in partnership with the California Department of Healthcare Services and has published on undercompensated subgroups in risk adjustment — the closest extant published comparable to our targeting study. Has cited our prior work on Medicaid coverage and heterogeneous treatment effects. Email: anna.zink@tufts.edu
3. **Stefan Feuerriegel, PhD** — LMU Munich, Head of the Institute of AI in Management. Co-author of the Manuf Serv Oper Manag 2024 paper on data-driven preventive-care allocation that we cite as the closest published precedent (ref 10), and of the Nature Medicine 2024 causal-ML review (ref 9). Has cited Basu's heterogeneous-treatment-effects work in the causal-ML-for-healthcare review literature. Email: feuerriegel@lmu.de
4. **Edward H. Kennedy, PhD** — Carnegie Mellon, Associate Professor of Statistics and Data Science. Author of the DR-Learner (ref 21) that we use as one of our principal benchmarks. *Disclosure: we cite his methodological work prominently; we are nonetheless suggesting him because he is the leading authority on whether the methodology is correctly applied.* Email: edward@stat.cmu.edu
5. **Iván Díaz, PhD** — NYU Grossman School of Medicine, Associate Professor of Population Health. Trained with Mark van der Laan; works on modern semiparametric causal inference for longitudinal data including longitudinal CATE — exactly the setting of our paper. Email: ivan.diaz@nyu.edu
6. **Kara E. Rudolph, PhD** — Columbia Mailman School of Public Health, Associate Professor of Epidemiology. Works on transporting treatment effects and identifying subpopulations most likely to benefit from interventions; has cited Basu's work on causal forests applied to clinical trial heterogeneity. Email: kr2854@cumc.columbia.edu

We have **no opposed reviewers**.

### Required statements

- This manuscript is original work; it has not been submitted to or accepted by any other journal in its current form. The earlier *PLOS Medicine* version, with its methodological errors, has been formally withdrawn from consideration there.
- All authors have read and approved the submitted manuscript.
- No part of this work has been published or presented elsewhere.
- Author contributions follow the CRediT taxonomy and are listed in the manuscript.
- Conflicts of interest are disclosed in the manuscript: all authors are employees of Waymark, the public-benefit organization whose Medicaid care-management programs provided the data; the funder had no role in study design, analysis, interpretation, or the decision to submit. The corresponding author also holds a UCSF faculty appointment.
- IRB approval: WCG IRB tracking ID 20253751, with waiver of informed consent for retrospective de-identified data analysis.
- Data availability: Individual-level data are not shareable per data-use agreements; aggregate code and synthetic-data reproduction are publicly available (URL above).
- AI-tool disclosure: Claude (Opus 4.7) was used for prose editing; GPT-4 for code scaffolding. No AI tool generated data, performed inference, or produced any results. No AI tool is listed as an author. Detailed disclosure is in the manuscript.

We are grateful for the opportunity to submit our work to *Population Health Management* and look forward to your review.

Sincerely,

**Sanjay Basu, MD, PhD**  
University of California, San Francisco  
Waymark  
sanjay.basu@ucsf.edu

---

**Attachments:** Manuscript (.docx), Supplementary Appendix (.docx), Title Page (.docx), Figures (3, .png 300 dpi), RECORD checklist (.pdf), TRIPOD+AI checklist (.pdf), ICMJE COI forms (.pdf, per author).
