# Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study

**Running Title:** Effect-Based Versus Risk-Based Care Management Targeting

## Authors

Sanjay Basu, MD, PhD$^{1,2}$; Sadiq Y. Patel, MSW, PhD$^{1,3}$; Rajaie Batniji, MD, PhD$^{1,4}$

$^{1}$Waymark, San Francisco, CA, USA  
$^{2}$Department of Epidemiology and Biostatistics, University of California, San Francisco, CA, USA  
$^{3}$School of Social Policy and Practice, University of Pennsylvania, Philadelphia, PA, USA  
$^{4}$Department of Medicine, Stanford University, Stanford, CA, USA

**Corresponding Author:** Sanjay Basu, MD, PhD. Email: sanjay.basu@ucsf.edu

**ORCID (corresponding author):** 0000-0002-7194-3458

**Word count, main text:** 4,478  
**Abstract word count:** 297  
**Figures:** 3 | **Tables:** 4 | **References:** 45

**Keywords:** Medicaid; care management; heterogeneous treatment effects; causal machine learning; population health management; health equity

---

## Abstract

**Background.** Medicaid care management programs typically allocate scarce outreach capacity to beneficiaries with the highest predicted risk of an acute event. This strategy implicitly assumes that risk and *responsiveness* to outreach are aligned across beneficiaries and stable across short time intervals. We tested whether targeting outreach by predicted *individualized treatment effect* — operationalized as the conditional average treatment effect (CATE) updated each calendar month — outperforms risk-based targeting in a contemporary Medicaid managed-care cohort, and whether any gains are equitably distributed.

**Methods.** Retrospective cohort study of 164,063 adult Medicaid beneficiaries (2,670,806 person-months) enrolled in community-based care management in Washington and Virginia, January 1, 2023 — December 31, 2025. The exposure was a completed care-manager telephone contact within a calendar month; the primary outcome was an emergency department visit or hospital admission within 30 days of the start of that month. CATEs were estimated using a causal forest with cross-fitted propensity-score and outcome-model nuisance functions, augmented by within-person fixed effects to control for time-invariant confounders. We compared two monthly allocation rules at 10% population capacity: a risk-based rule (top decile predicted event probability) and an effect-based rule (top decile predicted CATE). Policy values were estimated using doubly-robust off-policy evaluation. Variance in *estimated CATEs* was decomposed into between- and within-person components. Findings were validated through cross-state replication and a staggered-rollout instrumental-variable analysis with explicit pretrends, exclusion-restriction, and monotonicity diagnostics.

**Results.** Effect-based targeting prevented 13.3 (95% CI, 12.8 to 13.9) acute events per 2,000 members per month versus 2.5 (95% CI, 2.3 to 2.7) under risk-based targeting — a 5.3-fold (95% CI, 4.9 to 5.7) improvement. Within-person variance accounted for 63.6% (95% CI, 61.9 to 65.4) of total CATE variance. Effects were similar across racial/ethnic groups (effect-based vs. risk-based ratio: Black, 5.4×; White, 5.4×; Hispanic, 5.3×) and across states.

**Conclusions.** In two large Medicaid programs, monthly allocation of outreach by predicted treatment effect — rather than predicted risk — substantially increased the number of acute events prevented at fixed program capacity, without widening between-group disparities. Findings warrant prospective evaluation.

---

## Introduction

Medicaid care management programs operate under fixed labor capacity: a finite number of community health workers, care managers, and behavioral-health specialists must decide each month which beneficiaries to contact.$^{1,2}$ The dominant allocation heuristic is *risk-based targeting* — selecting beneficiaries with the highest predicted probability of an acute event such as an emergency department (ED) visit or hospital admission.$^{3,4}$ Risk-based targeting has the operational virtue of simplicity but rests on two implicit assumptions: that risk and treatment effect are highly correlated, and that an individual's responsiveness to outreach is approximately stable over the short intervals (typically one to three months) on which risk scores are recomputed.$^{5}$

Both assumptions are increasingly difficult to defend. Modern causal machine learning methods can estimate beneficiary-specific conditional average treatment effects (CATEs) from observational data,$^{6-9}$ and a growing body of empirical work shows that high *risk* and high *treatment effect* identify substantially nonoverlapping subsets of patients, with consequences for how preventive resources should be allocated.$^{10,11}$ A 2024 study of preventive diabetes care found that effect-based targeting roughly doubled the value of an intervention compared with risk-based targeting at equal capacity.$^{10}$ In parallel, longitudinal data from care management programs reveal that the same individual can move between states of higher and lower receptivity within weeks — for example, after a recent acute event, after a medication change, or following a period of contact non-response.$^{12,13}$

Despite this evidence, no published study has directly compared effect-based and risk-based monthly targeting in a contemporary Medicaid population using methods that meet current standards for causal-ML estimation, off-policy evaluation, and equity assessment.$^{14}$ Existing care-management evaluations rely overwhelmingly on risk scores; the few that estimate treatment effects do so cross-sectionally, ignoring within-person temporal variation; and most do not cross-validate against quasi-experimental designs.$^{4,15}$

We therefore conducted a retrospective cohort study in two state Medicaid programs to address three pre-specified questions. First, does monthly allocation of care-manager outreach by predicted CATE outperform allocation by predicted event probability at the same program capacity? Second, what fraction of variation in estimated CATEs occurs *within* individuals over time versus *between* individuals — that is, how much of the action lies in *when* to contact a given beneficiary versus *who* to contact? Third, are any efficiency gains from effect-based targeting distributed equitably across racial/ethnic and geographic subgroups? We anchored the analysis in the Reporting of studies Conducted using Observational Routinely-collected Data (RECORD) extension of STROBE,$^{16}$ pre-specified all primary contrasts, and validated findings using a staggered-rollout instrumental variable.

---

## Methods

### Study Design and Reporting

This was a retrospective cohort study using person-months as the unit of analysis. We followed RECORD$^{16}$ for routinely collected health data and TRIPOD+AI$^{17}$ for prediction model development. The study was approved by WCG Institutional Review Board (tracking ID 20253751) with waiver of informed consent. The analysis plan and primary contrasts were pre-specified before model fitting.

### Data Sources and Cohort

We used adjudicated medical and pharmacy claims, monthly enrollment files, and timestamped care-management encounter records from two Medicaid managed-care programs in Washington and Virginia, January 1, 2023 — December 31, 2025. Adults aged 18-64 years with continuous Medicaid managed-care eligibility (gaps ≤ 45 days) and ≥ 30 days of care-management enrollment during the study period were eligible. We excluded individuals who were Medicare-dual eligible, in hospice or long-term care facilities, or with conflicting enrollment records. The final analytic cohort contained 164,063 unique beneficiaries contributing 2,670,806 person-months. Cohort flow is reported in Figure S1.

### Exposure and Outcome

The unit of analysis was the person-month indexed by individual $i$ and calendar month $t$. The exposure $A_{it} \in \{0,1\}$ was a *completed* care-manager telephone contact during month $t$ (any non-zero number of completed contacts). To prevent reverse causation in which the outcome could precede the exposure within the same person-month, exposure was anchored to the first completed contact within month $t$, and the outcome window was defined as the 30 days *following* that anchor; person-months with an outcome event prior to any contact attempt were assigned to the no-exposure arm. Person-months without any contact attempt were anchored to day 1 of month $t$. The primary outcome was an ED visit or inpatient admission within the 30-day post-anchor window, identified from claims using bill type 11x or 12x for inpatient admissions and revenue codes 045x or 0981 for ED visits. Secondary outcomes were 90-day and 180-day events.

### Covariates

We constructed a 127-feature covariate vector $X_{it}$ at the start of each person-month, comprising: demographics (age, sex, race/ethnicity from enrollment, county-level Rural-Urban Commuting Area code, primary language, plan); historical utilization (counts and recency of ED, inpatient, observation, primary-care, specialist, behavioral-health, and dental encounters across 30/90/180/365-day windows); chronic-condition indicators from Chronic Conditions Data Warehouse algorithms; care-management engagement history (cumulative completed encounters, days since last completed encounter, days since last unsuccessful outreach attempt as a hard-to-reach proxy); pharmacy features (count of unique National Drug Codes, proportion of days covered for chronic medications, high-risk medication flags); and temporal features (calendar month, days since enrollment, days since last acute event). Race/ethnicity was self-reported and recorded by Medicaid agencies at enrollment; missingness (12.3%) was retained as an "Unknown" category, and a sensitivity analysis using multiple imputation by chained equations yielded equivalent estimates (Table S5). Continuous features were standardized; categorical features were one-hot encoded. The full covariate dictionary is in Appendix S1.2.

Forward-fill was used for utilization features (absence of a claim is informative — it represents no encounter). For static features missing within an individual, we used a hierarchy of (a) carry-forward from prior months, (b) within-state median, and tested robustness using multiple imputation by chained equations (Table S5).

### CATE Estimation

We estimated the conditional average treatment effect

$$\tau(x) = E[Y_i(1) - Y_i(0) \mid X_i = x]$$

where $Y_i(a)$ is the potential outcome under exposure $a \in \{0,1\}$. Identification required (i) consistency, (ii) conditional exchangeability $Y(0), Y(1) \perp A \mid X$, and (iii) positivity $0 < e(x) = \Pr(A=1 \mid X=x) < 1$. We discuss each assumption — and its threats — below and in Appendix S1.3.

The primary CATE estimator was a *causal forest* (generalized random forest with honest splitting),$^{18,19}$ fit to person-month data with $B = 4{,}000$ trees, minimum leaf size 100, mtry $= \lceil \sqrt{p} \rceil$, and an honest 50/50 sample split between leaf-construction and leaf-estimation samples. Causal forests directly target $\tau(x)$ rather than constructing it as the difference of two outcome models, and thus avoid the regularization-induced bias that plagues plug-in T-learner estimators in settings with heterogeneous treatment-effect signal.$^{20}$

For comparability we also fit (a) a T-learner (separate random-forest outcome models for treated and control); (b) an S-learner (single random-forest outcome model with treatment as a feature); (c) the *doubly robust learner* (DR-Learner) of Kennedy,$^{21}$ which constructs pseudo-outcomes $\widetilde{Y}_i = \mu_{A_i}(X_i) + (Y_i - \mu_{A_i}(X_i)) / \pi_{A_i}(X_i) \cdot (2A_i - 1)$ and regresses them on $X_i$ via gradient boosting; and (d) the R-learner of Nie and Wager.$^{22}$ All learners used the same nuisance functions $\hat\mu_a(x)$ (gradient-boosted regression of $Y$ on $X$ within $A=a$) and $\hat e(x)$ (gradient-boosted classifier for $A$ on $X$), fit using **5-fold cross-fitting** with sample-splitting between nuisance and target estimation to ensure $\sqrt{n}$-consistent inference per Chernozhukov et al.$^{23}$ Hyperparameters were selected by 5-fold cross-validation within the training fold, optimizing held-out negative log-likelihood for $\hat e$ and held-out mean-squared error for $\hat\mu_a$. The chosen tuning grids and selected values are in Appendix S1.3.

To address concerns about time-invariant unmeasured confounding (e.g., stable beneficiary preferences for engagement), we additionally specified a within-person fixed-effects extension by demeaning $X_{it}, A_{it}, Y_{it}$ by their person-specific time-averages prior to causal-forest fitting.$^{24,25}$ This identifies CATEs from variation *within* a beneficiary across months and removes any confounding by stable individual characteristics. We acknowledge — and return to in *Limitations* — that fixed effects do *not* address time-varying confounding (e.g., a clinical deterioration in month $t$ that simultaneously triggers outreach and elevates outcome risk).

To mitigate time-varying confounding within identifiable structure, we estimated stabilized inverse-probability-of-treatment weights (sIPTW) using a longitudinal propensity model with month-lagged covariates,$^{26,27}$ and re-estimated all primary contrasts under marginal structural model (MSM) targeting. We also conducted a target-trial emulation sensitivity analysis$^{28}$ in which exposure assignment was indexed at fixed monthly anchor times and follow-up was carried forward for 30 days regardless of subsequent enrollment changes.

### Allocation Rules and Off-Policy Evaluation

We compared two deterministic monthly allocation rules at fixed 10% capacity (the operational constraint of the Waymark care-management program):

- **Risk-based rule** $\pi_R$: each month, contact the top-decile beneficiaries by predicted 30-day event probability $\hat p(x)$ from a gradient-boosted classifier.
- **Effect-based rule** $\pi_E$: each month, contact the top-decile beneficiaries by predicted CATE $\hat\tau(x)$ from the causal forest.

Capacity is binding: each rule contacts the same number of beneficiaries each month. The behavior policy (the actual contact pattern observed in the data) was estimated as the cross-fitted propensity $\hat e(x)$.

Policy value was estimated by **doubly robust off-policy evaluation** (DR-OPE) of Jiang and Li,$^{29}$ which combines the direct method (model-based prediction of value) with importance sampling to achieve consistency when *either* the outcome model or the importance weights are correctly specified:

$$\hat V(\pi) = \frac{1}{N}\sum_i \left[ \hat\mu_{\pi(X_i)}(X_i) + \frac{\mathbb{1}[A_i = \pi(X_i)]}{\hat e(X_i)} (Y_i - \hat\mu_{A_i}(X_i)) \right].$$

The primary outcome metric was *acute events prevented per 2,000 members per month* under each policy. We computed 95% confidence intervals using cluster bootstrap with 1,000 resamples at the *individual* level (preserving within-person correlation) and stratified by state to preserve the joint Washington-Virginia distribution. The *efficiency ratio* $\hat V(\pi_E) / \hat V(\pi_R)$ was the primary contrast, with rank-weighted average treatment effect (RATE) curves of Yadlowsky et al.$^{30}$ as a secondary, capacity-free benchmark.

### Variance Decomposition of Estimated CATEs

To quantify how much heterogeneity in *predicted treatment effect* operates across individuals versus across time within an individual, we fit a person-level mixed model **on the estimated CATEs** $\hat\tau_{it}$ (not on the outcome $Y_{it}$):

$$\hat\tau_{it} = \alpha_i + u_{it}, \qquad \alpha_i \sim N(0, \sigma_B^2), \;\; u_{it} \sim N(0, \sigma_W^2).$$

We report the within-person fraction $\sigma_W^2 / (\sigma_B^2 + \sigma_W^2)$ with bootstrap 95% CIs. This formulation directly answers the question "how much CATE variation is within-person?" and corrects an error in our earlier analysis that decomposed *outcome* variance, conflating baseline-risk differences with treatment-effect differences.$^{31}$

### Quasi-Experimental Validation

We exploited staggered county-level rollout of the care-management contract as an instrument $Z_{it}$ for $A_{it}$. Staggered rollout was driven by contractual go-live dates negotiated with state agencies and was unrelated to individual beneficiary characteristics. We pre-registered three falsifiable diagnostics: (i) a **pretrends test** comparing month-by-month outcome rates in early- versus late-rollout counties for the 12 months *before* either rollout date, with no permitted divergence (slope difference 95% CI must include zero); (ii) an **exclusion-restriction probe** examining whether rollout was associated with any non-care-management exposure (provider-network changes, behavioral-health benefit changes, telehealth coverage) using county-level time-varying covariates; and (iii) a **monotonicity assessment** by comparing the sign of first-stage effects across pre-specified subgroups (defiers would manifest as opposite-signed first stages).$^{32}$ The IV estimator was 2SLS with cluster-robust standard errors at the county level. We additionally computed a Callaway-Sant'Anna staggered difference-in-differences estimator$^{33}$ to provide a non-IV benchmark.

### Equity Analysis

We pre-specified subgroup contrasts by self-reported race/ethnicity (Black or African American; White; Hispanic or Latino) and by state. For each subgroup we report (a) the absolute number of events prevented per 1,000 members per month under each rule; (b) the efficiency ratio; (c) the *allocation rate* (fraction of subgroup members contacted under each rule), with allocation parity defined as a Black-to-White rate ratio of 0.95-1.05; and (d) the *equalized-odds* difference in true-positive rate (CATE > 0 conditional on contact) across groups.

### Sensitivity Analyses

Pre-specified sensitivity analyses comprised: (a) E-value calculation for unmeasured confounding;$^{34}$ (b) propensity-score trimming at thresholds 0.01, 0.05, and 0.10; (c) alternative outcome models (gradient boosting, neural network, logistic regression); (d) alternative imputation (multiple imputation by chained equations); (e) longer outcome windows (90 and 180 days); (f) restriction to *non-deferrable* ED visits using the New York University ED algorithm$^{35}$ to address concerns that ED utilization may reflect access barriers rather than clinical need; (g) target-trial emulation; and (h) temporal split-sample validation training on 2023-2024 and evaluating on 2025.

### Sample Size and Statistical Inference

Sample size was determined by available data. We did not perform a post-hoc power calculation, which is uninformative once the analysis is complete and is mathematically a transformation of the observed *p*-value.$^{36}$ Instead, we report the precision of the primary efficiency ratio through its 95% CI and pre-specified an effect size of clinical interest (≥ 1.5×) for which the observed CI must exclude unity.

All analyses were conducted in Python 3.11 using the `econml` package (DR-Learner and forests, version 0.15.1), `grf` 2.4.0 (causal forest, called via `rpy2`), `scikit-learn` 1.4, `statsmodels` 0.14, and `numpy` 1.26. Statistical significance was defined as a two-sided 95% CI excluding the null. We did not adjust for multiple comparisons in subgroup analyses, which were pre-specified and tested the same primary hypothesis across strata.

### Data and Code Availability

Individual-level data cannot be shared due to data-use agreements with state Medicaid agencies and HIPAA. Aggregate summary statistics, all analysis code, container specifications, and a synthetic-data demonstration that reproduces every figure and table are publicly available at `https://github.com/sanjaybasu/medicaid-causal-world-model-` and archived with DOI in Zenodo. Researchers seeking individual-level data should contact the Washington State Health Care Authority and the Virginia Department of Medical Assistance Services through standard research-data-request procedures.

---

## Results

### Cohort

The 164,063 beneficiaries contributed 2,670,806 person-months over 36 months (Table 1). Mean age was 38.2 (SD 12.4) years; 60.5% were female; 41.2% identified as Black or African American, 42.3% as White, 8.9% as Hispanic or Latino. Across the cohort, beneficiaries received a median of 2 (IQR 0-4) completed care-manager contacts per year; 18.2% of person-months contained at least one completed contact, with 73.4% of those containing exactly one. The empirical capacity constraint was 9.7%-10.3% of members contacted per month, closely tracking the program's design target of 10%. Baseline 30-day acute-event rate was 7.9% across all person-months. State-stratified characteristics are in Table S1.

### Treatment-Effect Heterogeneity and Variance Decomposition

Across all 2,670,806 person-months, the mean estimated CATE from the cross-fitted causal forest was −0.025 (95% CI, −0.026 to −0.025), corresponding to an average 2.5 percentage-point absolute reduction in 30-day acute-event probability associated with a completed contact (Figure 2A). 87.8% of person-months had a CATE more favorable than zero (i.e., $\hat\tau < 0$); the lower decile had $\hat\tau \geq 0$, indicating either no benefit or the directional possibility of harm; we return to this finding under *Limitations*. (For ease of comparison with the risk-prediction literature, we report magnitudes in absolute risk-reduction units throughout, with negative CATE indicating event reduction.)

Decomposing variance in *estimated CATEs* (not in outcomes), the within-person component accounted for **63.6%** (95% CI, 61.9 to 65.4) of total variance; the between-person component, 36.4% (95% CI, 34.6 to 38.1) (Table 2). The same individual moved through estimated-CATE states whose standard deviation across months (median 0.031; IQR 0.019 to 0.046) was larger than the standard deviation of person-mean CATEs across the cohort (0.018). This finding reframes the targeting problem: at the modal individual, more uncertainty in *when* to contact lies within a beneficiary's longitudinal trajectory than across the population at any single moment.

### Effect-Based Versus Risk-Based Allocation

Effect-based allocation prevented **13.3 acute events per 2,000 members per month** (95% CI, 12.8 to 13.9) versus 2.5 (95% CI, 2.3 to 2.7) under risk-based allocation — an efficiency ratio of **5.3** (95% CI, 4.9 to 5.7) (Table 3, Figure 2B). Number-needed-to-contact was 15 (95% CI, 14 to 16) under the effect-based rule and 80 (95% CI, 74 to 87) under the risk-based rule.

Only 34.2% of beneficiaries selected by risk-based allocation in any given month were also selected by the effect-based rule that same month: high *risk* and high *effect* identified materially nonoverlapping populations, consistent with prior reports.$^{10,11}$ Among the 65.8% of effect-based selections not in the risk top-decile, the average baseline 30-day risk was 0.084 (vs. 0.291 for the risk-based selections) yet the average estimated CATE was −0.083 (vs. −0.013 for risk-based selections) — that is, the effect-based rule identified moderate-risk beneficiaries who, by their predicted CATE, had substantially more to gain from contact at that moment than higher-risk beneficiaries.

Sensitivity analyses (Table 4 and Table S5) showed the efficiency ratio was stable across CATE estimators (causal forest 5.3×; DR-Learner 5.2×; R-Learner 5.4×; T-Learner 4.0× — the lower T-Learner estimate consistent with regularization-bias attenuation); across propensity trimming thresholds (5.2× to 5.4×); under MSM/IPTW targeting (5.1×); under target-trial emulation (5.1×); and when restricting outcomes to non-deferrable ED visits per the NYU algorithm (4.9×). The temporal split-sample analysis trained on 2023-2024 and applied to held-out 2025 yielded an efficiency ratio of 5.0× (95% CI, 4.5 to 5.6), supporting the policy's stability over time. The E-value for the observed efficiency ratio was 8.7, meaning an unmeasured confounder would need to be associated with both contact assignment and outcome at relative-risk magnitude ≥ 8.7 to nullify the result; the strongest measured covariate (prior-90-day ED count) had a relative-risk of 2.3 with assignment and 3.1 with outcome, well below the E-value threshold.

The RATE curve (Figure 3A; Figure S6) was strictly concave with the area under the targeting curve significantly above the diagonal (AUTC = 0.31; 95% CI, 0.28 to 0.34), indicating that effect-based prioritization dominates risk-based prioritization at *every* operational capacity from 1% to 50% — not just at the 10% used in the program.

### Cross-State Replication

In Washington (n = 52,993), the efficiency ratio was 5.1 (95% CI, 4.5 to 5.8); in Virginia (n = 111,070), 5.4 (95% CI, 4.9 to 6.0); pooled by random-effects meta-analysis, 5.3 (95% CI, 4.8 to 5.9), with low heterogeneity ($I^2 = 8.2\%$, *P* = .31) (Table S2).

### Quasi-Experimental Validation

The staggered-rollout instrument had a strong first stage (effect on contact = 0.124; first-stage F = 127.4). Pre-trend testing showed no significant divergence in monthly outcome rates between early- and late-rollout counties in the 12 months prior to either county's rollout (slope difference −0.0004 per month; 95% CI, −0.0017 to 0.0009; Figure S4). The exclusion-restriction probe found no co-occurring policy or coverage changes correlated with rollout dates after adjustment for secular trends and state-level holidays (joint Wald test *P* = .27). The monotonicity check found same-signed first-stage effects across all pre-specified strata (Table S6).

The 2SLS local average treatment effect was −0.046 (95% CI, −0.049 to −0.042), corresponding to a 45.6% (95% CI, 42.1% to 49.0%) relative reduction in 30-day acute events among compliers (Table S3). The Callaway-Sant'Anna staggered DiD estimate was −0.039 (95% CI, −0.043 to −0.034), consistent with the IV estimate. Both quasi-experimental estimates lie within the bounds of the observational CATE distribution, supporting the credibility — though not the perfection — of identification.

### Equity

Efficiency ratios were similar across pre-specified subgroups (Table S7): Black beneficiaries, 5.4× (95% CI, 4.9 to 6.0); White, 5.4× (95% CI, 4.8 to 6.0); Hispanic, 5.3× (95% CI, 4.5 to 5.9); $P_{\text{interaction}} = 0.71$. Allocation rates were within the pre-specified parity band (Black-to-White rate ratio 0.99 [95% CI, 0.97 to 1.02]; Hispanic-to-White 0.99 [95% CI, 0.96 to 1.02]). Equalized-odds differences in true-positive rate across racial/ethnic groups were within 0.012 absolute, an order of magnitude below thresholds proposed in fairness-machine-learning practice.$^{37}$ State-level efficiency ratios were 5.4× (Virginia) and 5.1× (Washington), with overlapping confidence intervals.

---

## Discussion

In two large state Medicaid managed-care programs, replacing risk-based monthly targeting with allocation by predicted treatment effect — using a contemporary causal-machine-learning estimator with cross-fitted nuisance functions, doubly robust off-policy evaluation, and quasi-experimental validation — produced approximately a 5-fold increase in acute events prevented per 2,000 members per month at fixed program capacity. Within-person variation accounted for 63.6% of variance in *estimated treatment effects*. Effects were similar across racial/ethnic groups and states. These findings are consistent in direction and magnitude with the diabetes-prevention analysis of Kraus et al.,$^{10}$ extending that result from a single payer and disease into a multi-state, all-condition Medicaid care-management context.

The mechanism is intuitive in retrospect. Risk-based targeting concentrates outreach on beneficiaries whose elevated event probability is driven largely by stable characteristics — comorbidity, prior utilization, social determinants — many of which are not modifiable by a brief care-management contact. Effect-based targeting in contrast surfaces beneficiaries whose immediate clinical context has shifted them into a state where contact is plausibly consequential: a recent gap in chronic-disease medication, a missed primary-care visit, a transition from inpatient to home, or a documented engagement attempt that succeeded in a prior month. The fact that 64% of estimated-CATE variance is within-person reflects this temporal lability: the same beneficiary moves repeatedly between higher- and lower-receptivity states across months. Risk-stratification recomputed quarterly cannot capture this dynamic; CATE recomputed monthly can.

We emphasize what this study does *not* claim. We do not estimate true individual treatment effects (ITEs) — only conditional average treatment effects, which describe the average causal effect within a covariate stratum. The distinction is important: a beneficiary with a favorable CATE at month $t$ is, on average, the kind of beneficiary who benefits from outreach in that state, but the realized effect for any single individual may differ.$^{38}$ We also do not claim that observational identification is complete: even with cross-fitting, fixed effects, MSM/IPTW targeting, and a quasi-experimental check, we cannot rule out time-varying unmeasured confounding (for example, an in-month clinical deterioration that simultaneously increases the probability of a care-manager noticing and the probability of an event). The instrumental-variable analysis provides a triangulating estimate but rests on its own assumptions, which we tested and report transparently. A randomized adaptive trial of effect-based versus risk-based targeting is the natural next step.

The equity findings deserve attention. Risk-prediction algorithms have repeatedly been shown to misrank Black patients because risk scores incorporate access-driven utilization signals that themselves embed historical inequity.$^{39}$ Effect-based targeting is not immune to this concern — if outcome ascertainment differs by race, then estimated CATEs will inherit any systematic bias — but the within-person component of identification (each beneficiary serves as her own control over time) reduces the leverage of stable group-level confounding. We observed allocation parity, equalized true-positive rates, and similar efficiency ratios across racial/ethnic groups, none of which is guaranteed by the method itself. Programs adopting effect-based targeting should plan ongoing equity audits, including time-varying monitoring of allocation rates and outcomes by sub-group.

### Limitations

Several limitations warrant emphasis. First, the analysis is observational. Although we used cross-fitting, fixed effects, MSM/IPTW targeting, target-trial emulation, an instrumental-variable check, and an E-value of 8.7 to triangulate identification, residual time-varying confounding remains the most plausible threat to causal interpretation. Second, the outcome — ED visit or admission — is heterogeneous: some ED visits reflect access barriers rather than clinical need. Restricting to non-deferrable visits per the NYU algorithm$^{35}$ produced a slightly attenuated 4.9× efficiency ratio, supporting but not eliminating this concern. Third, ~12% of person-months had estimated CATEs $\geq 0$ (no benefit or possible harm). We interpret these as identification artifacts of finite-sample causal-forest estimates near a true effect of zero, exacerbated where the propensity is far from 0.5; we do not believe they reflect genuine outreach harm, but cannot exclude this possibility for individual beneficiaries.$^{40}$ Fourth, the study covers two states; though they differ in demographics, plan structure, and provider networks, generalization to other Medicaid programs, Medicare beneficiaries, or commercial populations should be confirmed empirically. Fifth, we evaluate a single intervention type — a completed care-manager contact. The relative efficiency of effect-based versus risk-based targeting may differ for higher-intensity interventions (home visits, complex case management) or non-clinical interventions (food assistance, housing support). Sixth, the program-fixed 10% capacity is operationally meaningful but not optimal; the RATE curve indicates effect-based dominance across all examined capacities, but optimization of *capacity itself* is outside scope.

### Implications

For Medicaid managed-care plans and state agencies, the practical implication is that the allocation logic of care management — embedded in dozens of risk-stratification tools currently in production — leaves substantial preventable acute care on the table. The infrastructure to replace risk-based scoring with effect-based scoring is moderate but real: monthly retraining of nuisance and CATE estimators, a model-monitoring system to detect drift, a model-card-style transparency artifact, and an equity-audit cadence. The policy implication is correspondingly direct: payment models that reward outreach *volume* among high-risk beneficiaries should be re-examined, since they incentivize contacts whose marginal effect is small. Payment models that reward outcomes — particularly outcomes among rotating, recomputed top-decile-by-CATE populations — would better align resource use with marginal value.

### Conclusions

Monthly allocation of Medicaid care-management outreach by predicted treatment effect, rather than by predicted risk, prevented approximately five times more acute events at fixed program capacity in two state Medicaid programs. The result was robust to multiple causal-ML estimators, sensitivity analyses, and a quasi-experimental check; was similar across racial/ethnic and geographic subgroups; and was driven primarily by within-person rather than between-person variation in estimated treatment effects. Prospective randomized evaluation is warranted.

---

## Tables

**Table 1. Cohort characteristics, by state.** *164,063 adult Medicaid beneficiaries; 2,670,806 person-months; January 1, 2023 — December 31, 2025.*

| Characteristic | Overall (n = 164,063) | Virginia (n = 111,070) | Washington (n = 52,993) |
|---|---|---|---|
| Age, years, mean (SD) | 38.2 (12.4) | 38.5 (12.3) | 37.6 (12.6) |
| Female, n (%) | 99,258 (60.5) | 67,227 (60.5) | 32,031 (60.4) |
| Black or African American, n (%) | 67,594 (41.2) | 52,341 (47.1) | 15,253 (28.8) |
| White, n (%) | 69,399 (42.3) | 43,890 (39.5) | 25,509 (48.1) |
| Hispanic or Latino, n (%) | 14,602 (8.9) | 8,885 (8.0) | 5,717 (10.8) |
| Other or unknown, n (%) | 12,468 (7.6) | 5,954 (5.4) | 6,514 (12.3) |
| Prior ED visits per person-year, mean (SD) | 0.83 (1.42) | 0.86 (1.45) | 0.77 (1.36) |
| Prior hospitalizations per person-year, mean (SD) | 0.21 (0.58) | 0.22 (0.60) | 0.19 (0.54) |
| ≥ 1 chronic condition, n (%) | 128,625 (78.4) | 87,423 (78.7) | 41,202 (77.7) |
| ≥ 3 chronic conditions, n (%) | 69,071 (42.1) | 47,184 (42.5) | 21,887 (41.3) |
| Documented behavioral-health diagnosis, n (%) | 51,844 (31.6) | 35,482 (31.9) | 16,362 (30.9) |
| Person-months contributed | 2,670,806 | 1,808,515 | 862,291 |
| Median completed contacts per person-year (IQR) | 2 (0–4) | 2 (0–4) | 2 (0–4) |
| Person-months with ≥ 1 completed contact, % | 18.2 | 18.5 | 17.5 |
| 30-day ED-or-admission rate, % of person-months | 7.9 | 8.1 | 7.5 |

---

**Table 2. Variance decomposition of estimated conditional average treatment effects.** *Mixed-effects model fit on cross-fitted causal-forest CATE estimates, with person-level random intercepts.*

| Variance component | Estimate | 95% CI | Fraction of total |
|---|---|---|---|
| Between-person, $\sigma_B^2$ | $3.24 \times 10^{-4}$ | (3.07, 3.41) $\times 10^{-4}$ | 36.4% (34.6 to 38.1) |
| Within-person, $\sigma_W^2$ | $5.66 \times 10^{-4}$ | (5.51, 5.82) $\times 10^{-4}$ | 63.6% (61.9 to 65.4) |
| Total | $8.90 \times 10^{-4}$ | — | 100% |

*Decomposition is on estimated treatment effects $\hat\tau_{it}$, correcting an earlier formulation that decomposed the outcome $Y_{it}$.*

---

**Table 3. Primary contrast: effect-based vs. risk-based allocation at 10% capacity.**

| Allocation rule | Acute events prevented per 2,000 members per month (95% CI) | Number needed to contact (95% CI) | Mean CATE among contacted (95% CI) |
|---|---|---|---|
| Risk-based ($\pi_R$) | 2.5 (2.3 to 2.7) | 80 (74 to 87) | −0.013 (−0.014 to −0.012) |
| Effect-based ($\pi_E$) | 13.3 (12.8 to 13.9) | 15 (14 to 16) | −0.067 (−0.070 to −0.064) |
| **Efficiency ratio** $\pi_E / \pi_R$ | **5.3 (4.9 to 5.7)** | — | — |

*CIs from individual-level cluster bootstrap, 1,000 replicates, stratified by state. Negative CATE = absolute risk reduction.*

---

**Table 4. Sensitivity analyses.**

| Sensitivity domain | Specification | Efficiency ratio (95% CI) |
|---|---|---|
| Primary | Causal forest, cross-fit, 10% capacity | 5.3 (4.9 to 5.7) |
| CATE estimator | DR-Learner (Kennedy 2023) | 5.2 (4.7 to 5.7) |
| | R-Learner (Nie & Wager 2021) | 5.4 (4.9 to 5.9) |
| | T-Learner | 4.0 (3.5 to 4.5) |
| | S-Learner | 4.6 (4.1 to 5.2) |
| Time-varying confounding | MSM with stabilized IPTW | 5.1 (4.6 to 5.6) |
| Identification design | Target-trial emulation | 5.1 (4.6 to 5.7) |
| Outcome restriction | Non-deferrable ED only (NYU algorithm) | 4.9 (4.4 to 5.5) |
| Outcome window | 90-day events | 4.7 (4.3 to 5.2) |
| | 180-day events | 4.3 (3.9 to 4.7) |
| Propensity trimming | 0.01 | 5.2 (4.7 to 5.6) |
| | 0.10 | 5.4 (4.9 to 5.9) |
| Imputation | Multiple imputation by chained equations | 5.2 (4.8 to 5.7) |
| Temporal validation | Train 2023-2024, test 2025 | 5.0 (4.5 to 5.6) |
| E-value | Required RR for nullification | 8.7 |

---

## Figure Legends

**Figure 1. Analytic schema.** Each calendar month $t$, every beneficiary is described by a 127-feature covariate vector $X_{it}$ including demographics, prior utilization, chronic-condition indicators, care-management engagement history, and temporal markers. Cross-fitted nuisance functions estimate the propensity $\hat e(x) = \Pr(A=1 \mid X=x)$ and outcome regressions $\hat\mu_a(x) = E[Y \mid A=a, X=x]$. A causal forest combines these into the conditional average treatment effect $\hat\tau(x) = E[Y(1) - Y(0) \mid X=x]$. Two monthly allocation rules are compared at fixed 10% capacity: $\pi_R$ contacts the top-decile by predicted risk $\hat p(x) = \hat\mu_0(x)$; $\pi_E$ contacts the top-decile by predicted CATE $\hat\tau(x)$. Policy values are estimated via doubly robust off-policy evaluation. Within-person fixed effects extension demeans $X_{it}, A_{it}, Y_{it}$ by person-specific time-averages prior to forest fitting to control for time-invariant confounders.

**Figure 2. Distribution of estimated CATEs and primary contrast.** *(A)* Histogram of 2,670,806 cross-fitted causal-forest CATE estimates. The mean of −0.025 corresponds to a 2.5-percentage-point absolute reduction in 30-day acute-event probability. 87.8% of person-months had $\hat\tau < 0$. The lower decile, which clusters near zero, is the population most likely to reflect identification artifact rather than true zero or harmful effects; we discuss interpretation in *Limitations*. *(B)* Acute events prevented per 2,000 members per month under risk-based ($\pi_R$, blue) and effect-based ($\pi_E$, orange) allocation, both at fixed 10% capacity. Bars are point estimates from doubly robust off-policy evaluation; whiskers are 95% confidence intervals from 1,000-replicate individual-level cluster bootstrap.

**Figure 3. Capacity-free contrast and quasi-experimental validation.** *(A)* Rank-weighted average treatment effect (RATE) curve. The x-axis is operational capacity $k$ from 1% to 50%; the y-axis is mean CATE among the top-$k$ contacted beneficiaries. The effect-based rule (orange) lies strictly above the risk-based rule (blue) across all examined capacities. The area under the targeting curve (AUTC) is 0.31 (95% CI, 0.28 to 0.34). *(B)* Quasi-experimental validation: 30-day acute-event rate by predicted-CATE tertile in the IV-eligible subsample (n = 24,997 beneficiaries straddling staggered rollout). Higher predicted-CATE tertiles experienced lower outcome rates (low: 46.2%; medium: 27.2%; high: 25.1%; *P* < .001), consistent with the observational CATE estimates and supporting external validity.

---

## Acknowledgments

We thank the Medicaid beneficiaries whose care contributed to this analysis; the community health workers, care managers, behavioral-health specialists, and pharmacists who delivered care; the Washington State Health Care Authority and Virginia Department of Medical Assistance Services for data access; and Waymark data engineering and analytics teams for data curation and infrastructure. We thank Drs. Sherri Rose and James Robins for methodological discussion, and the four PLOS Medicine reviewers whose detailed feedback substantially improved the methods.

## Funding

Internal Waymark research funds. The funder had no role in study design, analysis, interpretation, manuscript preparation, or the decision to submit.

## Conflicts of Interest

S.B., S.Y.P., and R.B. are employees of Waymark, a public-benefit organization that delivers Medicaid care management. S.B. is also faculty at UCSF. The findings reflect the authors' independent analysis; Waymark had no editorial control.

## Author Contributions

**Conceptualization:** S.B. **Methodology:** S.B. **Software:** S.B. **Formal analysis:** S.B. **Data curation:** S.B. **Writing — original draft:** S.B. **Writing — review & editing:** S.B., S.Y.P., R.B. **Visualization:** S.B. **Supervision:** S.Y.P., R.B.

## AI Tool Disclosure

We used Anthropic's Claude (Opus 4.7) as a writing aid for prose editing and to draft figure-legend wording, and OpenAI's GPT-4 to generate boilerplate code scaffolding. All analytic code, model specifications, results, citations, and final text were authored, verified, and edited by the human authors. No AI tool generated data, performed inference, or produced any results reported here. No AI tool is listed as an author per ICMJE policy.

---

## References

1. Berwick DM, Nolan TW, Whittington J. The triple aim: care, health, and cost. *Health Aff (Millwood)*. 2008;27(3):759-769.

2. Bodenheimer T, Sinsky C. From triple to quadruple aim: care of the patient requires care of the provider. *Ann Fam Med*. 2014;12(6):573-576.

3. Billings J, Blunt I, Steventon A, Georghiou T, Lewis G, Bardsley M. Development of a predictive model to identify inpatients at risk of re-admission within 30 days of discharge (PARR-30). *BMJ Open*. 2012;2(4):e001667.

4. Wallace E, Stuart E, Vaughan N, Bennett K, Fahey T, Smith SM. Risk prediction models to predict emergency hospital admission in community-dwelling adults: a systematic review. *Med Care*. 2014;52(8):751-765.

5. Bates DW, Saria S, Ohno-Machado L, Shah A, Escobar G. Big data in health care: using analytics to identify and manage high-risk and high-cost patients. *Health Aff (Millwood)*. 2014;33(7):1123-1131.

6. Athey S, Imbens G. Recursive partitioning for heterogeneous causal effects. *Proc Natl Acad Sci U S A*. 2016;113(27):7353-7360.

7. Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. *J Am Stat Assoc*. 2018;113(523):1228-1242.

8. Künzel SR, Sekhon JS, Bickel PJ, Yu B. Metalearners for estimating heterogeneous treatment effects using machine learning. *Proc Natl Acad Sci U S A*. 2019;116(10):4156-4165.

9. Feuerriegel S, Frauen D, Melnychuk V, et al. Causal machine learning for predicting treatment outcomes. *Nat Med*. 2024;30(4):958-968.

10. Kraus M, Feuerriegel S, Saar-Tsechansky M. Data-driven allocation of preventive care with application to diabetes mellitus type II. *Manuf Serv Oper Manag*. 2024;26(1):137-153.

11. Bertsimas D, Kallus N. From predictive to prescriptive analytics. *Manage Sci*. 2020;66(3):1025-1044.

12. Lei H, Nahum-Shani I, Lynch K, Oslin D, Murphy SA. A "SMART" design for building individualized treatment sequences. *Annu Rev Clin Psychol*. 2012;8:21-48.

13. Coleman EA, Boult C; American Geriatrics Society Health Care Systems Committee. Improving the quality of transitional care for persons with complex care needs. *J Am Geriatr Soc*. 2003;51(4):556-557.

14. Naimi AI, Mishler AE, Kennedy EH. Challenges in obtaining valid causal effect estimates with machine learning algorithms. *Am J Epidemiol*. 2023;192(9):1536-1544.

15. Haas LR, Takahashi PY, Shah ND, et al. Risk-stratification methods for identifying patients for care coordination. *Am J Manag Care*. 2013;19(9):725-732.

16. Benchimol EI, Smeeth L, Guttmann A, et al; RECORD Working Committee. The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement. *PLoS Med*. 2015;12(10):e1001885.

17. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.

18. Athey S, Tibshirani J, Wager S. Generalized random forests. *Ann Stat*. 2019;47(2):1148-1178.

19. Tibshirani J, Athey S, Sverdrup E, Wager S. grf: Generalized random forests. R package version 2.4.0. 2024.

20. Curth A, van der Schaar M. Nonparametric estimation of heterogeneous treatment effects: from theory to learning algorithms. *Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)*. PMLR. 2021;130:1810-1818.

21. Kennedy EH. Towards optimal doubly robust estimation of heterogeneous causal effects. *Electron J Stat*. 2023;17(2):3008-3049.

22. Nie X, Wager S. Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*. 2021;108(2):299-319.

23. Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. *Econom J*. 2018;21(1):C1-C68.

24. Imai K, Kim IS. When should we use unit fixed effects regression models for causal inference with longitudinal data? *Am J Pol Sci*. 2019;63(2):467-490.

25. Imai K, Kim IS. On the use of two-way fixed effects regression models for causal inference with panel data. *Polit Anal*. 2021;29(3):405-415.

26. Robins JM, Hernán MA, Brumback B. Marginal structural models and causal inference in epidemiology. *Epidemiology*. 2000;11(5):550-560.

27. Hernán MA, Robins JM. *Causal Inference: What If*. Boca Raton, FL: Chapman & Hall/CRC; 2020.

28. Hernán MA, Robins JM. Using big data to emulate a target trial when a randomized trial is not available. *Am J Epidemiol*. 2016;183(8):758-764.

29. Jiang N, Li L. Doubly robust off-policy value evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. PMLR. 2016;48:652-661.

30. Yadlowsky S, Fleming S, Shah N, Brunskill E, Wager S. Evaluating treatment prioritization rules via rank-weighted average treatment effects. *J Am Stat Assoc*. 2025;120(549):38-51.

31. Vegetabile BG. On the distinction between "conditional average treatment effects" (CATE) and "individual treatment effects" (ITE) under ignorability assumptions. arXiv:2108.04939. 2021.

32. Imbens GW, Angrist JD. Identification and estimation of local average treatment effects. *Econometrica*. 1994;62(2):467-475.

33. Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time periods. *J Econom*. 2021;225(2):200-230.

34. VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*. 2017;167(4):268-274.

35. Billings J, Parikh N, Mijanovich T. Emergency department use: the New York story. *Issue Brief (Commonw Fund)*. 2000;(434):1-12.

36. Althouse AD. Post hoc power: not empowering, just misleading. *J Surg Res*. 2021;259:A3-A6.

37. Rajkomar A, Hardt M, Howell MD, Corrado G, Chin MH. Ensuring fairness in machine learning to advance health equity. *Ann Intern Med*. 2018;169(12):866-872.

38. Powers S, Qian J, Jung K, et al. Some methods for heterogeneous treatment effect estimation in high dimensions. *Stat Med*. 2018;37(11):1767-1787.

39. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. *Science*. 2019;366(6464):447-453.

40. Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. *Biometrics*. 2005;61(4):962-973.

41. Hahn PR, Murray JS, Carvalho CM. Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects. *Bayesian Anal*. 2020;15(3):965-1056.

42. Chakraborty B, Murphy SA. Dynamic treatment regimes. *Annu Rev Stat Appl*. 2014;1:447-464.

43. Murphy SA. Optimal dynamic treatment regimes. *J R Stat Soc Series B Stat Methodol*. 2003;65(2):331-355.

44. Pourat N, Zhou W, Haley LA, et al. Health Resources and Services Administration-funded health centers reduce health care expenditures of California Medicaid managed care beneficiaries with complex needs. *Popul Health Manag*. 2025;28(3):117-124.

45. Sakinah I, Bertozzi L, Patel S, et al. Additive impact of virtual urgent and emergency department at home care on value-based primary care for Medicaid and dual-eligible members. *Popul Health Manag*. 2025;28(3):173-178.
