"""
Medicaid Causal World Model package.

Provides scaffolding for:
- Stitching claims, encounters, and ADT into timelines.
- Extracting structured states and latent confounders from text.
- Fitting causal structures and sequential policies (Causal Decision Transformers).
- Running counterfactual simulations and offline evaluation.
"""

__all__ = [
    "data_schema",
    "state_extraction",
    "causal_world_model",
    "policies",
    "evaluation",
]

__version__ = "0.1.0"
