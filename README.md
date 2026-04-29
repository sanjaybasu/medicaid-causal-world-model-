# Medicaid Causal Machine Learning Targeting Study

This repository contains code and submission materials for:

**"Treatment-Effect-Based Versus Risk-Based Targeting of Care Management Outreach in Medicaid: A Causal Machine Learning Cohort Study"**

*Submitted to Population Health Management (April 2026)*

The earlier version of this manuscript was evaluated at PLOS Medicine and not accepted; the present version was substantially rebuilt to address every reviewer concern. See [`submission_phm_revision/README.md`](submission_phm_revision/README.md) for a point-by-point summary.

## Key findings

- 164,063 Medicaid beneficiaries, 2,670,806 person-months in Washington and Virginia (Jan 2023 – Dec 2025)
- Effect-based monthly allocation prevented **5.3× more acute events** than risk-based allocation at fixed 10% capacity
- Within-person variance accounted for **63.6%** of total variation in estimated treatment effects
- Efficiency gains were similar across racial/ethnic and geographic subgroups
- Findings validated via cross-state replication, MSM/IPTW, target-trial emulation, and a staggered-rollout instrumental variable with explicit pretrend, exclusion-restriction, and monotonicity diagnostics

## Repository layout

- [`submission_phm_revision/`](submission_phm_revision/) — final manuscript, appendix, cover letter, figures (.docx and .md)
- [`code/`](code/) — analysis pipeline (build event table → CATE estimation → policy evaluation → figures)
- [`outputs/`](outputs/) — JSON outputs from the production runs
- [`src/medicaid_causal_world_model/`](src/medicaid_causal_world_model/) — Python package implementing the analysis primitives
- [`expected_outputs/`](expected_outputs/) — expected outputs for verification
- [`tests/`](tests/) — unit tests
- [`REPLICATION.md`](REPLICATION.md) — step-by-step replication instructions

## Citation

```bibtex
@article{basu2026effecttargeting,
  title   = {Treatment-Effect-Based Versus Risk-Based Targeting of Care
             Management Outreach in Medicaid: A Causal Machine Learning
             Cohort Study},
  author  = {Basu, Sanjay and Patel, Sadiq Y and Batniji, Rajaie},
  journal = {Population Health Management},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT (see [LICENSE](LICENSE)).

## Contact

Sanjay Basu, MD, PhD — University of California, San Francisco, and Waymark — sanjay.basu@ucsf.edu
