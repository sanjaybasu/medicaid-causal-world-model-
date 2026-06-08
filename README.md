# Medicaid Causal Machine Learning Targeting Study — Code

Reproducible analysis code for:

**Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study**

## What the code does

The pipeline (in [`code/`](code/)) implements:

1. **Event-table construction** — stitches Medicaid claims, monthly enrollment, and care-management encounter records into a person-month panel.
2. **Feature engineering** — builds the 127-feature covariate vector $X_{it}$ described in the manuscript Methods (demographics, historical utilization, chronic conditions, engagement history, pharmacy, temporal features).
3. **Nuisance estimation** — cross-fitted (5-fold) propensity-score and outcome-model nuisance functions via gradient boosting, with isotonic-regression calibration of propensities.
4. **CATE estimation** — primary causal forest (`grf` v2.4.0) with honest splitting; benchmark estimators include the DR-Learner (Kennedy 2023), R-Learner (Nie & Wager 2021), T-Learner, and S-Learner.
5. **Within-person fixed effects extension** — demeans covariates, treatment, and outcome by person-specific time-averages prior to causal-forest fitting.
6. **Identification triangulation** — marginal structural model with stabilized IPTW, target-trial emulation, and staggered-rollout instrumental-variable analysis with pretrend, exclusion-restriction, and monotonicity diagnostics.
7. **Off-policy evaluation** — doubly-robust value estimator (Jiang & Li 2016) for both risk-based and effect-based monthly allocation rules at fixed 10% capacity.
8. **Variance decomposition** — mixed model on cross-fitted CATE estimates partitioning into between- and within-person components.
9. **Equity diagnostics** — subgroup efficiency ratios, allocation parity, equalized-odds.
10. **Sensitivity analyses** — alternative CATE estimators, propensity trimming, imputation, outcome models, outcome windows, restriction to non-deferrable ED visits (NYU algorithm), temporal split-sample validation, E-value.

## Repository layout

- [`code/`](code/) — numbered pipeline scripts (`01_build_event_table.py` → `12_hybrid_fe_world_model.py`)
- [`src/medicaid_causal_world_model/`](src/medicaid_causal_world_model/) — Python package with reusable primitives (data schema, state extraction hooks, policy scoring, evaluation utilities)
- [`tests/`](tests/) — unit tests
- [`pyproject.toml`](pyproject.toml) — package definition and dependencies
- [`REPLICATION.md`](REPLICATION.md) — step-by-step replication instructions including expected runtime, hardware, and external data dependencies
- [`LICENSE`](LICENSE) — MIT
- [`.gitignore`](.gitignore) — enforces the code-only policy: data, outputs, manuscripts, and figures are explicitly excluded

## Citation

```bibtex
@article{basu2026effecttargeting,
  title   = {Treatment-Effect-Based Versus Risk-Based Targeting of Care
             Management Outreach in Medicaid: A Causal Machine Learning
             Cohort Study},
  author  = {Basu, Sanjay and Patel, Sadiq Y and Batniji, Rajaie},
  year    = {2026}
}
```


## License

MIT (see [LICENSE](LICENSE)).

## Contact

Sanjay Basu, MD, PhD — University of California, San Francisco, and Waymark — sanjay.basu@ucsf.edu
