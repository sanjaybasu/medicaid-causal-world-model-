# Medicaid Causal World Model

[![GitHub](https://img.shields.io/badge/GitHub-medicaid--causal--world--model-blue)](https://github.com/sanjaybasu/medicaid-causal-world-model)

This repository contains code and materials for:

**"Temporal Optimization of Population Health Interventions Using Causal World Models: A Multi-State Medicaid Study"**


## Overview

This package implements a counterfactual simulator for Medicaid members, combining within-person fixed effects causal inference with deep generative world modeling. It upgrades prior work (`healthcare-world-model`, `engagement_analysis` v5) by adding latent confounders from encounters text and sequential policy learning over claims + ADT + encounter timelines.

## Scope
- **Data stitching:** unify eligibility, claims, encounters text/actions, and ADT events into an ordered event stream with censoring flags.
- **Perception:** LLM prompts + DeepSCM to extract structured SDOH and latent confounders from notes.
- **Causal structure:** LLM-proposed DAG validated with DECI/DAG-GNN constraints.
- **Dynamics + policy:** Causal Decision Transformer for offline policy scoring and counterfactual rollouts.
- **Evaluation:** target-trial emulation replay, IV sensitivity, offline OPE, fairness checks.

## Layout
- `src/medicaid_causal_world_model/`
  - `data_schema.py` — typed event/state containers for stitched timelines.
  - `state_extraction.py` — hooks for LLM tagging and DeepSCM training.
  - `causal_world_model.py` — SCM + CDT orchestration and counterfactual simulation.
- `policies.py` — policy scoring and recommendation surfaces for multidisciplinary teams (CHWs, therapists, pharmacists, care coordinators).
- `taxonomy.py` — intervention/SDOH/role taxonomy and keyword bootstrapper for encounter notes.
- `timeline.py` — helpers to convert encounters (and later claims/ADT) into sorted patient timelines.
  - `evaluation.py` — offline evaluation utilities (DR/OPE, fairness).
- `tests/` — placeholders for unit and integration tests.
- `pyproject.toml` — simple editable install.

## Quick start
1. `pip install -e .` from this directory.
2. Implement timeline construction feeding `data_schema.PatientEvent`.
3. Add LLM prompt templates and DeepSCM training in `state_extraction.py`, then persist latent states.
4. Fit causal structure and CDT via `causal_world_model.py`; evaluate with `evaluation.py`.

Coordinate experiments and reporting with the notebook plan in `notebooks/medicaid_causal_world_model/`.

---

## Manuscript Submission

The `submission/` directory contains final materials for PLOS Medicine:
- Main manuscript and supplementary appendix
- Three publication-quality figures (300 DPI)
- Cover letter and verification documentation

**Key findings:**
- 164,063 Medicaid beneficiaries, 2,842,718 person-months (2022-2025)
- Dynamic allocation: **5.3× more efficient** than risk-based allocation
- 64% of treatment effect heterogeneity occurs within-person over time
- Validated with natural experiment (p < 0.001)

See `submission/README.md` for details.

---

## Citation

If you use this code or reference the manuscript, please cite:

```bibtex
@article{basu2026temporal,
  title={Temporal Optimization of Population Health Interventions Using Causal World Models: A Multi-State Medicaid Study},
  author={Basu, Sanjay and Patel, Sadiq Y and Batniji, Rajaie},
  journal={Submitted to PLOS Medicine},
  year={2026}
}
```

## License

Code released under MIT License. See LICENSE file for details.

## Contact

For questions about the code or manuscript:
- Sanjay Basu, MD, PhD
- Waymark & University of California San Francisco
