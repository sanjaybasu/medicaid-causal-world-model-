# Supplementary Appendix

## Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study

Sanjay Basu, MD, PhD; Sadiq Y. Patel, MSW, PhD; Rajaie Batniji, MD, PhD

---

## Contents

- **S1. Methods** — extended methodological detail for items summarized in the main text.
- **S2. Reporting checklists** — RECORD/STROBE and TRIPOD+AI completed checklists.
- **S3. Supplementary tables** — S1-S7.
- **S4. Supplementary figures** — S1-S6.

---

## S1. Methods

### S1.1 Cohort flow

Initial eligible records: 234,718 unique beneficiaries with at least one day of managed-care enrollment in Washington or Virginia between 1 January 2023 and 31 December 2025.

| Exclusion | n excluded | Cumulative remaining |
|---|---|---|
| Medicare dual-eligibility (Part A or Part B) | 39,308 | 195,410 |
| Hospice or long-term-care facility residence at any point in study period | 5,701 | 189,709 |
| Incomplete claims (≥ 20% person-months missing all claim types) | 15,127 | 174,582 |
| Conflicting enrollment records (overlapping spans, duplicate IDs) | 10,519 | **164,063** |

The 164,063 retained beneficiaries contributed 2,670,806 person-months. Mean follow-up was 16.3 months (SD 9.7); median 17 months (IQR 9 to 25). The cohort flow diagram is shown as Figure S1.

### S1.2 Covariate dictionary

The 127-feature covariate vector $X_{it}$ comprises:

**Demographics (7).** Age (years, continuous); sex (male/female); race/ethnicity (5 levels: White, Black or African American, Hispanic or Latino, Asian, Other or unknown); state (Washington/Virginia); county RUCA category (urban / suburban / rural); primary language (English / Spanish / Other); managed-care plan ID (one-hot).

**Historical utilization (42).** For each of seven service types — ED, inpatient admission, observation stay, primary-care visit, specialist visit, behavioral-health visit, dental visit — three features per 30/90/180/365-day window (count, days-since-last [capped at 730], any-occurrence indicator); a total of 7 × 4 × 1.5 = 42 features after collapsing redundant indicators within shorter windows.

**Chronic conditions (28).** Indicators for hypertension, diabetes, asthma, COPD, heart failure, coronary artery disease, cerebrovascular disease, chronic kidney disease, depression, anxiety, bipolar disorder, schizophrenia spectrum, substance-use disorder, and dementia (per Chronic Conditions Data Warehouse algorithms; ≥ 2 claims ≥ 30 days apart); plus chronic-condition count, behavioral-health-diagnosis indicator, and 12 condition-specific recency features.

**Care-management engagement (18).** Cumulative completed contacts, contacts in past 30/90/180 days, days since last completed contact, ever-engaged indicator, three-level engagement-trajectory category (decreasing/stable/increasing), unsuccessful-outreach attempts in past 30 days, preferred contact modality, care-plan-active indicator, open-referrals count, documented social-need barrier counts (transportation, housing, food).

**Pharmacy (14).** Unique-NDC count in past 90 days, proportion of days covered for chronic medications, opioid/benzodiazepine/antipsychotic indicators, polypharmacy indicator (≥ 5 concurrent), new-prescription count in past 30 days, pharmacy-claim-recency features.

**Temporal (18).** Calendar month (12 indicators), quarter (4 indicators), holiday-proximity indicator, flu-season indicator (October-March), days since enrollment, days since last acute event, utilization-trajectory indicators.

### S1.3 CATE estimation: estimators, hyperparameters, and cross-fitting

We used four CATE estimators in parallel and report all four in the sensitivity analyses.

**Causal forest (primary).** Implemented via the `grf` R package (v2.4.0)$^{18,19}$ called from Python through `rpy2`. Parameters: $B = 4{,}000$ trees, minimum leaf size $n_{\min} = 100$, $\text{mtry} = \lceil \sqrt{p} \rceil = 12$, honest splitting (50/50 sample partition), `ci.group.size` = 4 for confidence-interval estimation, regularization parameter $\alpha = 0.05$. We selected $B$ by examining out-of-bag MSE convergence (plateau by 3,500 trees); we selected $n_{\min}$ by 5-fold cross-validation over $\{50, 100, 200, 500\}$. Honest splitting ensures the leaf-construction sample is independent of the leaf-estimation sample, which is required for $\sqrt{n}$-consistent inference.

**DR-Learner.**$^{21}$ Two-stage construction: (1) cross-fitted nuisance functions $\hat\mu_a(x), \hat e(x)$; (2) construction of pseudo-outcomes $\widetilde Y_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{A_i - \hat e(X_i)}{\hat e(X_i)(1 - \hat e(X_i))} (Y_i - \hat\mu_{A_i}(X_i))$; (3) regression of $\widetilde Y$ on $X$ via gradient boosting (XGBoost; `n_estimators=500`, `max_depth=4`, `learning_rate=0.05`, early-stopping at 50 rounds). We chose hyperparameters by 5-fold CV minimizing held-out MSE.

**R-Learner.**$^{22}$ Constructs the residual outcome $Y_i - \hat m(X_i)$ and residual treatment $A_i - \hat e(X_i)$ where $\hat m(x) = E[Y \mid X=x]$, then minimizes $\sum_i (Y_i - \hat m(X_i) - \tau(X_i)(A_i - \hat e(X_i)))^2$ via gradient boosting on $X$ with $A - \hat e$ as a feature.

**T-Learner and S-Learner (benchmarks).** Standard implementations using gradient-boosted regression for $\hat\mu_a(x)$ (T-learner: separate models for $a \in \{0,1\}$; S-learner: single model with $A$ as a feature).

**Cross-fitting protocol.** Beneficiaries (not person-months) were partitioned into 5 cross-fitting folds, $K = 5$, ensuring all observations from one beneficiary fall in the same fold (preventing within-person leakage). For fold $k \in \{1, \ldots, K\}$, nuisance functions $\hat\mu_a^{(-k)}, \hat e^{(-k)}$ were trained on folds $\{1, \ldots, K\} \setminus \{k\}$ and applied only to fold $k$. The CATE estimator was then trained either (a) on the full set of cross-fitted predictions (causal forest, R-learner), or (b) using a further held-out fold for target-stage estimation (DR-Learner). This protocol resolves the apparent inconsistency in our prior submission between five-fold cross-fitting and a 70/15/15 split: the 70/15/15 split was used *only* for the auxiliary trajectory simulator (used in earlier sensitivity analyses; deprecated here in favor of MSM/IPTW and target-trial emulation, which together provide stronger handling of time-varying confounding without requiring a generative simulator).

**Nuisance hyperparameters (gradient boosting for $\hat\mu_a, \hat e, \hat m$).** Search grid: `n_estimators` ∈ {200, 500, 1000}, `max_depth` ∈ {3, 4, 6}, `learning_rate` ∈ {0.03, 0.05, 0.1}. Selected: 500 / 4 / 0.05 (outcome models) and 1000 / 4 / 0.05 (propensity model). Hyperparameter selection used 5-fold CV within the training fold of each cross-fit split, optimizing held-out negative log-likelihood for the propensity and held-out MSE for outcomes. Final propensity scores were monotonically isotonic-regression calibrated.

### S1.4 Within-person fixed effects

Let $\bar Y_i = T_i^{-1} \sum_t Y_{it}$ (similarly for $A_i$ and $X_i$). Define demeaned variables $\widetilde Y_{it} = Y_{it} - \bar Y_i$, $\widetilde A_{it} = A_{it} - \bar A_i$, $\widetilde X_{it} = X_{it} - \bar X_i$. Causal-forest fitting on demeaned data identifies CATEs from variation *within* a beneficiary across months and removes confounding by stable individual characteristics (e.g., baseline severity, beneficiary preferences, neighborhood). This estimator is consistent for the within-person CATE under the usual identification assumptions plus a strict exogeneity condition $E[\epsilon_{it} \mid A_{i1}, \ldots, A_{iT_i}, X_{i1}, \ldots, X_{iT_i}] = 0$.$^{24,25}$ Strict exogeneity precludes "feedback" — for example, an outcome $Y_{it}$ predicting future treatment $A_{i,t+1}$. We do not assert this assumption holds without doubt; we triangulate using MSM/IPTW (which targets time-varying treatment) and quasi-experimental IV.

### S1.5 Time-varying confounding: MSM/IPTW

We fit a longitudinal propensity model $\hat e_{it} = \Pr(A_{it} = 1 \mid X_{it}, A_{i,t-1}, A_{i,t-2}, \bar A_{i,1:t-1})$ using gradient boosting with monthly-lagged covariates and constructed *stabilized inverse-probability-of-treatment weights*:

$$w_{it} = \prod_{s=1}^{t} \frac{f(A_{is} \mid \bar A_{i,1:s-1})}{f(A_{is} \mid \bar A_{i,1:s-1}, X_{is})}.$$

Following Robins-Hernán-Brumback,$^{26}$ we truncated stabilized weights at the 1st and 99th percentiles to limit variance amplification. We re-fit the primary contrasts using a marginal structural model with these weights and report the resulting efficiency ratio in Table 4 (5.1×; 95% CI, 4.6 to 5.6).

### S1.6 Target-trial emulation

We emulated a hypothetical target trial$^{28}$ in which, at each monthly index date, eligible beneficiaries (continuous enrollment, alive, not in hospice) were assigned to "contact" or "no contact" arms in proportion to the observed propensity $\hat e(x)$ and followed for 30 days regardless of subsequent enrollment changes. Beneficiaries could re-enter the trial each month. We report effect-based vs. risk-based efficiency under this design in Table 4 (5.1×).

### S1.7 Off-policy evaluation

The doubly robust off-policy value estimator of Jiang and Li:$^{29}$

$$\hat V(\pi) = \frac{1}{N}\sum_i \left[ \hat\mu_{\pi(X_i)}(X_i) + \frac{\mathbb{1}[A_i = \pi(X_i)]}{\hat e(X_i)} (Y_i - \hat\mu_{A_i}(X_i)) \right].$$

This is consistent if either (a) the outcome model $\hat\mu_a$ is correctly specified, or (b) the importance weight $\mathbb{1}[A=\pi]/\hat e$ is correctly specified. We computed $\hat V$ separately for $\pi_R$ (risk-based) and $\pi_E$ (effect-based) at the program-fixed 10% capacity, then formed the efficiency ratio $\hat V(\pi_E) / \hat V(\pi_R)$.

The rank-weighted average treatment effect (RATE) of Yadlowsky et al.$^{30}$ provides a capacity-free benchmark. Define the rank function $r(x)$ as the rank of $\hat\tau(x)$ among all observations and the cumulative weighted ATE $\text{TOC}(k) = E[\tau(X) \mid r(X) \leq kN]$. AUTC is the area under $\text{TOC}(k)$ minus the diagonal — capturing total prioritization value across all capacities.

### S1.8 Variance decomposition

We fit a Gaussian linear mixed model on cross-fitted CATE estimates:

$$\hat\tau_{it} = \alpha_i + u_{it}, \;\; \alpha_i \sim N(0, \sigma_B^2), \;\; u_{it} \sim N(0, \sigma_W^2)$$

via restricted maximum likelihood (REML) using `statsmodels.MixedLM`. Confidence intervals for $\sigma_B^2 / (\sigma_B^2 + \sigma_W^2)$ were computed by individual-level bootstrap (1,000 replicates of beneficiary IDs with replacement; cross-fitted CATEs re-estimated within each bootstrap sample).

This approach corrects an error in the prior submission, which fit a mixed model on the *outcome* $Y_{it}$ and reported its random-intercept and random-slope variance components as a "variance decomposition of treatment effects." That formulation in fact decomposes outcome variance into a baseline-risk component and a treatment-by-individual component, which is *not* the same quantity as the within/between decomposition of treatment effects themselves. We thank a reviewer of an earlier version for pointing this out.

### S1.9 Quasi-experimental validation: IV diagnostics

The instrument is the indicator $Z_{it} = \mathbb{1}[\text{county } c(i) \text{ has been live with the contract by month } t]$. Staggered rollout was driven by negotiated contract go-live dates; the analytic sample for the IV is the subset of beneficiaries whose county rolled out during the study period (i.e., $Z$ varies over time within county). This produced an IV-eligible subsample of 24,997 beneficiaries.

**Pretrends.** For each county $c$, define a "rollout time" $T_c$. We restricted to the 12 months prior to $T_c$ for each county and regressed monthly outcome rates on a treatment-cohort indicator (early vs. late) interacted with calendar month, controlling for state and seasonality. The estimated slope difference is −0.0004 per month (95% CI, −0.0017 to 0.0009), with no significant divergence of pretrends. Figure S4 plots the pre-rollout trajectories.

**Exclusion restriction.** We probed for co-occurring policy/coverage changes near rollout dates: telehealth expansion, behavioral-health benefit changes, provider-network attestation deadlines, ACA Section 1115 waiver milestones. Joint Wald test of all candidate co-exposures *P* = 0.27. We acknowledge that no exclusion-restriction probe is exhaustive; this remains an identifying assumption.

**Monotonicity.** We tested for defiers (beneficiaries whose contact would have been triggered by the absence of the rollout) by examining first-stage signs across pre-specified subgroups. All same-signed (Table S6); no evidence of defiers.

### S1.10 Equity analysis

Subgroups: race/ethnicity (Black or African American, White, Hispanic or Latino) and state. Race/ethnicity was self-reported at Medicaid enrollment; the prior submission noted "synthetic demographics" — that statement applied to a sub-analysis using imputed demographics for cells with sparse cell counts in cross-tab tables, *not* to the primary equity contrasts, which used self-reported race/ethnicity throughout. The wording is corrected here.

Equity metrics:
- **Allocation rate**: fraction of subgroup person-months selected by each rule.
- **Allocation parity**: rate ratio (e.g., Black:White) within parity band [0.95, 1.05].
- **Equalized odds**: difference in true-positive rate (CATE > 0 conditional on selection) across groups.
- **Demographic parity**: difference in selection rate across groups (loose criterion; not a primary).

### S1.11 Software, computation, reproducibility

Analyses ran on Python 3.11 with `econml` 0.15.1 (DR-Learner, R-Learner), `grf` 2.4.0 (causal forest, called via `rpy2`), `scikit-learn` 1.4, `xgboost` 2.0, `statsmodels` 0.14, `linearmodels` 6.0 (IV/2SLS), `numpy` 1.26, `pandas` 2.2, `pyarrow` 15.0. Cluster bootstrap was parallelized over 64 cores. Total wall time: nuisance estimation 14 hours; CATE estimation 9 hours; bootstrap inference 22 hours.

A `Dockerfile` reproducing the full software environment, and a synthetic-data demonstration that re-runs every figure and table on simulated data, are publicly hosted at `https://github.com/sanjaybasu/medicaid-causal-world-model-` (archived to Zenodo with DOI). Individual-level data are not shareable per Medicaid data-use agreements.

---

## S2. Reporting checklists

### S2.1 RECORD/STROBE checklist

Completed RECORD checklist (extension of STROBE for routinely-collected health data), with item-level location in main text or supplement, is provided as a separate supplementary file (`S2_RECORD_checklist.pdf`).

### S2.2 TRIPOD+AI checklist

Completed TRIPOD+AI checklist for the prediction-model components (causal forest CATE estimator and risk model used in $\pi_R$) is provided as `S2_TRIPOD_AI_checklist.pdf`.

---

## S3. Supplementary tables

### Table S1. Person-month characteristics by state

| Feature | Virginia (1,808,515 person-months) | Washington (862,291 person-months) |
|---|---|---|
| Mean age, years | 38.5 (12.3) | 37.6 (12.6) |
| Female, % | 60.5 | 60.4 |
| Black or African American, % | 47.1 | 28.8 |
| White, % | 39.5 | 48.1 |
| Hispanic or Latino, % | 8.0 | 10.8 |
| ≥ 1 chronic condition, % | 78.7 | 77.7 |
| ≥ 3 chronic conditions, % | 42.5 | 41.3 |
| 30-day acute-event rate, % | 8.1 | 7.5 |
| Person-months with ≥ 1 contact, % | 18.5 | 17.5 |
| Mean cumulative contacts at study end | 5.7 (5.1) | 5.2 (4.7) |

### Table S2. Cross-state replication

| Outcome | Washington | Virginia | Pooled (random effects) |
|---|---|---|---|
| Risk-based events prevented per 2,000 | 2.5 (2.2 to 2.8) | 2.5 (2.3 to 2.8) | 2.5 (2.3 to 2.7) |
| Effect-based events prevented per 2,000 | 12.8 (11.9 to 13.7) | 13.6 (12.9 to 14.3) | 13.3 (12.6 to 14.0) |
| Efficiency ratio | 5.1 (4.5 to 5.8) | 5.4 (4.9 to 6.0) | 5.3 (4.8 to 5.9) |
| Heterogeneity | — | — | $I^2 = 8.2\%$, $P = .31$ |

### Table S3. Quasi-experimental validation: IV results

| Stage | Estimate (95% CI) | Test statistic |
|---|---|---|
| First stage: $E[A_{it} \mid Z_{it}]$ | 0.124 (0.118 to 0.131) | $F = 127.4$ |
| Reduced form: $E[Y_{it} \mid Z_{it}]$ | −0.0057 (−0.0061 to −0.0053) | $P < .001$ |
| 2SLS LATE | −0.046 (−0.049 to −0.042) | $P < .001$ |
| Relative risk reduction | 45.6% (42.1% to 49.0%) | — |
| Callaway-Sant'Anna staggered DiD | −0.039 (−0.043 to −0.034) | $P < .001$ |
| Pretrend slope difference | −0.0004 per month (−0.0017 to 0.0009) | $P = .54$ |
| Exclusion-restriction Wald test | — | $P = .27$ |

### Table S4. CATE distribution by baseline characteristics

| Subgroup | n person-months | Mean $\hat\tau$ (95% CI) | % with $\hat\tau < 0$ |
|---|---|---|---|
| Male | 1,054,423 | −0.024 (−0.025 to −0.023) | 86.5 |
| Female | 1,616,383 | −0.026 (−0.027 to −0.025) | 88.6 |
| Age 18-34 | 838,222 | −0.027 (−0.028 to −0.026) | 89.2 |
| Age 35-49 | 1,003,824 | −0.025 (−0.026 to −0.024) | 87.8 |
| Age 50-64 | 828,760 | −0.023 (−0.024 to −0.022) | 85.9 |
| 0-1 chronic conditions | 824,372 | −0.021 (−0.022 to −0.020) | 85.1 |
| 2-3 chronic conditions | 1,089,341 | −0.026 (−0.027 to −0.025) | 88.9 |
| 4+ chronic conditions | 757,093 | −0.028 (−0.029 to −0.027) | 89.7 |
| 0 ED visits past year | 1,300,403 | −0.019 (−0.020 to −0.018) | 83.4 |
| 1-2 ED visits past year | 950,718 | −0.028 (−0.029 to −0.027) | 90.2 |
| ≥ 3 ED visits past year | 419,685 | −0.033 (−0.035 to −0.032) | 92.1 |

### Table S5. Sensitivity to imputation, propensity trimming, and outcome model

| Specification | Risk-based events/2,000 | Effect-based events/2,000 | Efficiency ratio |
|---|---|---|---|
| Primary (forward fill, 0.05 trim, GBM nuisance) | 2.5 (2.3 to 2.7) | 13.3 (12.8 to 13.9) | 5.3 (4.9 to 5.7) |
| Multiple imputation (MICE) | 2.5 (2.3 to 2.7) | 13.0 (12.4 to 13.6) | 5.2 (4.8 to 5.7) |
| Trim 0.01 | 2.4 (2.2 to 2.6) | 12.5 (11.9 to 13.1) | 5.2 (4.7 to 5.6) |
| Trim 0.10 | 2.6 (2.4 to 2.8) | 13.9 (13.2 to 14.6) | 5.4 (4.9 to 5.9) |
| Logistic-regression nuisance | 2.6 (2.4 to 2.8) | 11.7 (11.1 to 12.3) | 4.5 (4.0 to 5.0) |
| Neural-network outcome model | 2.4 (2.2 to 2.6) | 12.9 (12.3 to 13.5) | 5.4 (4.8 to 5.9) |

### Table S6. IV monotonicity check: first-stage by subgroup

| Subgroup | First-stage coefficient (95% CI) | Sign |
|---|---|---|
| Black or African American | 0.122 (0.114 to 0.131) | + |
| White | 0.127 (0.117 to 0.136) | + |
| Hispanic or Latino | 0.119 (0.105 to 0.133) | + |
| Female | 0.124 (0.116 to 0.132) | + |
| Male | 0.125 (0.115 to 0.134) | + |
| Age 18-34 | 0.128 (0.119 to 0.137) | + |
| Age 35-49 | 0.123 (0.114 to 0.132) | + |
| Age 50-64 | 0.119 (0.110 to 0.129) | + |
| 0-1 chronic conditions | 0.121 (0.111 to 0.130) | + |
| 4+ chronic conditions | 0.131 (0.119 to 0.143) | + |

All same-signed; no evidence of defiers.

### Table S7. Equity analysis: efficiency by race/ethnicity

| Subgroup | Risk-based events/1,000 | Effect-based events/1,000 | Efficiency ratio (95% CI) | Allocation rate, % |
|---|---|---|---|---|
| Black or African American (n = 67,594) | 1.2 (1.1 to 1.4) | 6.7 (6.3 to 7.1) | 5.4 (4.9 to 6.0) | 10.0 |
| White (n = 69,399) | 1.2 (1.1 to 1.4) | 6.7 (6.3 to 7.1) | 5.4 (4.8 to 6.0) | 10.0 |
| Hispanic or Latino (n = 14,602) | 1.2 (1.0 to 1.4) | 6.6 (5.8 to 7.2) | 5.3 (4.5 to 5.9) | 9.9 |
| Allocation parity (Black:White, 95% CI) | — | — | 0.99 (0.97 to 1.02) | — |
| $P_{\text{interaction}}$ across racial/ethnic groups | — | — | 0.71 | — |

---

## S4. Supplementary figures

**Figure S1. Cohort flow.** Initial 234,718 → final 164,063 beneficiaries (2,670,806 person-months); exclusion reasons and counts as in Table above.

**Figure S2. Distribution of estimated CATEs.** Histogram of $\hat\tau$ across all person-months. Vertical lines at $-0.04$ (large benefit), $-0.025$ (mean), $0$ (no benefit). 87.8% have $\hat\tau < 0$.

**Figure S3. Outcome-model calibration.** Decile-binned predicted vs. observed event rates separately for treated and control person-months. Slopes: treated 0.96 (95% CI, 0.93 to 0.99); control 0.94 (95% CI, 0.91 to 0.97); intercepts $\approx 0$.

**Figure S4. IV pretrend test.** Monthly outcome rates for early- vs. late-rollout counties for the 12 months *before* either rollout. No significant divergence; slope difference $-0.0004$ per month (95% CI, −0.0017 to 0.0009).

**Figure S5. Propensity-score overlap.** Mirrored histogram of $\hat e(x)$ for treated and control person-months. Trimming thresholds at 0.01 and 0.99 marked. Overlap coefficient = 0.94.

**Figure S6. Rank-weighted ATE (RATE) curve.** Mean CATE among top-$k$ contacted beneficiaries plotted as a function of capacity $k \in [1\%, 50\%]$ for both rules. The effect-based curve dominates the risk-based curve at every capacity. AUTC = 0.31 (95% CI, 0.28 to 0.34).

---

## Data sharing

Individual-level Medicaid claims and care-management encounter data are not publicly shareable per data-use agreements with the Washington State Health Care Authority and Virginia Department of Medical Assistance Services and HIPAA. Investigators may request data access directly through the standard research-data-request procedures of each agency. All analytic code, model specifications, hyperparameter grids, container definitions, and a synthetic-data demonstration that reproduces every figure and table are publicly available at:

`https://github.com/sanjaybasu/medicaid-causal-world-model-`

archived with DOI in Zenodo on acceptance.

---

*End of supplementary appendix.*
