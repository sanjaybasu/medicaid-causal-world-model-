"""
Orchestration for the Medicaid Causal World Model.

Responsibilities:
- Fit/validate a causal graph (SCM) over stitched state variables.
- Train a sequential policy model (Causal Decision Transformer or similar).
- Run counterfactual rollouts for candidate interventions and return uplift + uncertainty.
"""

from typing import Any, Dict, List, Sequence

import pandas as pd

from .data_schema import PatientEvent, PatientState, PolicyRecommendation, Timeline


class CausalWorldModel:
    """Wrapper that combines SCM structure, dynamics, and policy scoring."""

    def __init__(self, scm: Any = None, policy_model: Any = None):
        self.scm = scm
        self.policy_model = policy_model

    def fit_structure(
        self,
        events: pd.DataFrame,
        dag_prior: Dict[str, Sequence[str]],
        constraints: Dict[str, Any],
    ) -> Any:
        """
        Fit or validate a causal graph using data and LLM/DAG priors.

        Args:
            events: flattened event-level dataset with covariates, actions, outcomes.
            dag_prior: adjacency suggestions from LLM/domain experts.
            constraints: temporal or forbidden edges to enforce acyclicity.
        """
        raise NotImplementedError("Implement DECI/DAG-GNN fitting with priors and constraints.")

    def fit_policy(
        self,
        timelines: List[Timeline],
        reward_horizon_days: int = 180,
        grace_window_days: int = 14,
        **kwargs: Any,
    ) -> Any:
        """
        Train a sequential policy model (e.g., Causal Decision Transformer) on timelines.

        Args:
            timelines: list of member timelines sorted by time with attached latent_z.
            reward_horizon_days: horizon for ADT avoidance.
            grace_window_days: enforce action masking to avoid immortal-time bias.
            kwargs: extra hyperparameters for the transformer/trainer.
        """
        raise NotImplementedError("Train CDT or related sequential model on patient timelines.")

    def recommend_actions(
        self,
        state: PatientState,
        candidate_actions: Sequence[str],
        num_rollouts: int = 50,
    ) -> List[PolicyRecommendation]:
        """
        Run counterfactual rollouts for candidate actions and return uplift + uncertainty.
        """
        raise NotImplementedError("Use fitted SCM + policy model to simulate interventions.")

    def simulate_timeline(
        self,
        timeline: Timeline,
        policy: Any,
        num_rollouts: int = 20,
    ) -> Dict[str, Any]:
        """
        Simulate a full future trajectory for a given policy.

        Returns:
            Dict with predicted outcomes, cumulative reward, and per-step actions.
        """
        raise NotImplementedError("Simulate forward trajectories under a given policy.")
