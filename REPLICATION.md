# Replication Guide

This repository contains code and materials to replicate the analysis in:

**"Temporal Optimization of Population Health Interventions Using Causal World Models: A Multi-State Medicaid Study"**

*Submitted to PLOS Medicine (January 2026)*

---

## Repository Structure

```
medicaid_causal_world_model/
│
├── README.md                   # Overview and quick start
├── REPLICATION.md             # This file - detailed replication guide
├── pyproject.toml             # Python package configuration
├── requirements.txt           # Python dependencies
│
├── src/                       # Python package (installable)
│   └── medicaid_causal_world_model/
│       ├── causal_world_model.py    # Main model implementation
│       ├── data_prep.py             # Data preprocessing
│       ├── data_schema.py           # Data structures
│       ├── evaluation.py            # Evaluation utilities
│       ├── offline_rl.py            # Offline RL methods
│       ├── policies.py              # Policy implementations
│       ├── state_extraction.py      # Feature engineering
│       ├── taxonomy.py              # Intervention taxonomy
│       └── timeline.py              # Timeline utilities
│
├── code/                      # Analysis pipeline scripts
│   ├── 01_build_event_table.py           # Create event timelines
│   ├── 02_extract_interventions.py       # Parse care management data
│   ├── 03_train_rl_models.py             # Train RL models
│   ├── 04_evaluate_policies.py           # Off-policy evaluation
│   ├── 05_compute_treatment_effects.py   # CATE estimation
│   ├── 06_train_static_risk_model.py     # Risk model baseline
│   ├── 07_generate_figures.py            # Create figures
│   ├── 08_compile_submission.py          # Compile manuscript
│   ├── 09_within_person_analysis.py      # Variance decomposition
│   ├── 10_system_optimization.py         # Policy optimization
│   ├── 11_package_submission.py          # Package for submission
│   ├── 12_hybrid_fe_world_model.py       # Main analysis
│   ├── config.yaml                       # Configuration
│   └── requirements.txt                  # Dependencies
│
├── outputs/                   # Analysis results (JSON files)
│   ├── hybrid_results.json                    # Primary outcomes
│   ├── within_person_variance.json            # Variance decomposition
│   ├── natural_experiment_validation.json     # Validation results
│   ├── atc_optimization.json                  # Policy optimization
│   ├── world_model_results.json               # Model diagnostics
│   └── verified_citations.json                # Reference verification
│
├── submission/                # PLOS Medicine submission materials
│   ├── PLOS_Medicine_Manuscript_FINAL_CORRECTED.md
│   ├── PLOS_Medicine_Supplementary_Appendix_FINAL_CORRECTED.md
│   ├── PLOS_Medicine_Cover_Letter_FINAL.md
│   ├── figures/
│   │   ├── Figure_1_Architecture.png          # 300 DPI
│   │   ├── Figure_2_Trajectories.png          # 300 DPI
│   │   └── Figure_3_Impact_Validation.png     # 300 DPI
│   └── create_figure3_two_panel.py            # Figure 3 generation
│
└── tests/                     # Unit tests (placeholder)
```

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 16 GB RAM minimum (32 GB recommended)
- 50 GB free disk space
- Linux, macOS, or Windows with WSL

### Python Dependencies

Install the package and dependencies:

```bash
# Clone repository
git clone https://github.com/sanjaybasu/medicaid-causal-world-model.git
cd medicaid-causal-world-model

# Install in editable mode
pip install -e .

# Or install dependencies manually
pip install -r code/requirements.txt
```

**Key dependencies:**
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `torch>=1.10.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`

---

## Data Requirements

**NOTE:** Patient-level data cannot be shared due to HIPAA and data use agreements.

The analysis requires three data sources:

1. **Medicaid Claims Data**
   - Medical and pharmacy claims
   - Diagnosis codes (ICD-10-CM)
   - Procedure codes (CPT, HCPCS)
   - Monthly granularity

2. **Care Management Encounter Records**
   - Encounter dates and types
   - Outreach attempts
   - Completed encounters
   - Intervention types

3. **Enrollment Files**
   - Member demographics
   - Enrollment dates
   - Plan information
   - County/state

### Data Format

Required data structure (example schema):

```python
# Event table (output of 01_build_event_table.py)
events = pd.DataFrame({
    'member_id': str,           # Hashed member identifier
    'month_year': datetime,     # Calendar month
    'treatment': int,           # 1 if intervention, 0 otherwise
    'outcome': int,             # 1 if acute event, 0 otherwise
    'age': int,
    'sex': str,
    'state': str,               # 'VA' or 'WA'
    # ... additional features
})
```

### Simulated Data

For testing/demonstration purposes, use simulated data:

```bash
python code/00_generate_synthetic_data.py --n-members 1000 --output data/synthetic/
```

**WARNING:** Results from simulated data will not match manuscript results.

---

## Replication Steps

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
python -c "import medicaid_causal_world_model; print('Success!')"
```

### Step 2: Configure Analysis

Edit `code/config.yaml`:

```yaml
data:
  input_dir: "data/real_inputs/"      # Your data location
  output_dir: "outputs/"

study:
  start_date: "2023-01-01"
  end_date: "2025-12-31"
  states: ["VA", "WA"]

model:
  random_seed: 42
  n_bootstrap: 1000
```

### Step 3: Run Analysis Pipeline

Execute scripts in order:

```bash
# 1. Build event table
python code/01_build_event_table.py

# 2. Extract interventions
python code/02_extract_interventions.py

# 3-11. Continue pipeline
python code/03_train_rl_models.py
python code/04_evaluate_policies.py
# ... (continue through step 12)

# 12. Main analysis (hybrid FE + world model)
python code/12_hybrid_fe_world_model.py
```

**Or run entire pipeline:**

```bash
bash code/run_pipeline.sh
```

### Step 4: Generate Figures

```bash
# Generate all manuscript figures
python code/07_generate_figures.py

# Generate Figure 3 specifically
python submission/create_figure3_two_panel.py
```

Outputs saved to `submission/figures/`

### Step 5: Verify Results

Compare your results to the published outputs:

```bash
python code/verify_replication.py
```

Expected outputs in `outputs/`:
- `hybrid_results.json` - Primary outcomes match manuscript Table 2
- `within_person_variance.json` - Variance decomposition (64% within-person)
- `natural_experiment_validation.json` - Validation results (p < 0.001)

---

## Key Analysis Steps

### Within-Person Fixed Effects

File: `code/09_within_person_analysis.py`

Estimates conditional average treatment effects (CATEs) using within-person fixed effects:

```python
# Simplified example
from medicaid_causal_world_model import causal_world_model

model = causal_world_model.CausalWorldModel()
model.fit(data, method='fixed_effects')
cates = model.predict_cates(data)
```

Key equation:
```
CATE_it = E[Y_it(1) - Y_it(0) | X_it]
```

### Deep Generative World Model

File: `code/12_hybrid_fe_world_model.py`

Two-stage approach:
1. **Stage 1 (Teacher):** Within-person FE for unbiased CATE estimates
2. **Stage 2 (Student):** Deep neural network learns to predict CATEs from features

```python
# Stage 1: Causal identification
teacher_model.fit(data, member_fixed_effects=True)
unbiased_cates = teacher_model.predict(data)

# Stage 2: Generative modeling
student_model.fit(data, teacher_targets=unbiased_cates)
predicted_cates = student_model.predict(new_data)
```

### Policy Evaluation

File: `code/04_evaluate_policies.py`

Off-policy evaluation using doubly robust estimation:

```python
from medicaid_causal_world_model import evaluation

# Define policies
risk_based_policy = lambda x: x['baseline_risk'] > threshold
dynamic_policy = lambda x: x['predicted_cate'] > threshold

# Evaluate
results = evaluation.off_policy_evaluation(
    data,
    policies={'risk': risk_based_policy, 'dynamic': dynamic_policy},
    capacity_constraint=0.10  # 10% of population
)
```

---

## Expected Results

### Primary Outcomes (from `hybrid_results.json`)

```json
{
  "risk_policy_events_prevented": 2.511,
  "atc_policy_events_prevented": 13.326,
  "efficiency_gain": 5.307,
  "mean_cate": 0.0250,
  "std_cate": 0.0235
}
```

### Variance Decomposition (from `within_person_variance.json`)

```json
{
  "within_person_variance": 0.636,
  "between_person_variance": 0.364,
  "total_variance": 0.0006
}
```

### Validation (from `natural_experiment_validation.json`)

```json
{
  "natural_experiment": {
    "p_value": 9.19e-165,
    "relative_reduction": 0.456,
    "validation_result": "PASSED"
  }
}
```

---

## Computational Requirements

### Runtime Estimates

With 164,063 beneficiaries and 2,842,718 person-months:

| Step | Runtime | Memory | Notes |
|------|---------|--------|-------|
| 01. Build event table | 2-4 hours | 8 GB | Depends on data format |
| 02. Extract interventions | 30-60 min | 4 GB | |
| 05. Compute CATEs | 4-6 hours | 16 GB | Most intensive step |
| 09. Within-person analysis | 2-3 hours | 12 GB | Bootstrap iterations |
| 12. Hybrid FE + WM | 6-8 hours | 16 GB | Neural network training |
| **Total** | **~20-30 hours** | **16 GB peak** | Single-threaded |

**With parallelization:** Can reduce to 8-12 hours using 4-8 cores.

### GPU Acceleration

Optional but recommended for faster training:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Expected speedup: 2-4× for neural network training steps.

---

## Troubleshooting

### Common Issues

**1. Memory errors during CATE estimation**

Solution: Process in batches

```python
# In code/05_compute_treatment_effects.py
batch_size = 10000  # Reduce if needed
```

**2. Convergence issues in RL training**

Solution: Adjust hyperparameters

```yaml
# In code/config.yaml
model:
  learning_rate: 0.0001  # Reduce if unstable
  n_epochs: 100          # Increase if underfitting
```

**3. Different results than manuscript**

Check:
- Random seed set to 42
- Same Python/package versions
- Data preprocessing steps identical
- Bootstrap iterations (n=1000)

---

## Verification Checklist

After running the full pipeline, verify:

- [ ] `outputs/hybrid_results.json` exists
- [ ] Efficiency gain ≈ 5.3× (within 5% tolerance)
- [ ] Within-person variance ≈ 64% (within 2% tolerance)
- [ ] P-value < 0.001 for natural experiment
- [ ] Figure 3 shows 2 panels (A, B)
- [ ] All 3 figures generated at 300 DPI

---

## Citation

If you use this code, please cite:

```bibtex
@article{basu2026temporal,
  title={Temporal Optimization of Population Health Interventions Using Causal World Models: A Multi-State Medicaid Study},
  author={Basu, Sanjay and Patel, Sadiq Y and Batniji, Rajaie},
  journal={Submitted to PLOS Medicine},
  year={2026}
}
```

---

## Support

For questions about replication:

1. Check existing issues: https://github.com/sanjaybasu/medicaid-causal-world-model/issues
2. Open new issue with:
   - Python version
   - Error message
   - Steps to reproduce
3. Contact: Sanjay Basu (sbasu@waymark.com)

---

## License

Code: MIT License (see LICENSE file)

Data: Not available (HIPAA restrictions)

Manuscript: © Authors, submitted to PLOS Medicine

---

**Last Updated:** January 31, 2026
**Version:** 1.0
**Status:** Replication ready with synthetic data
