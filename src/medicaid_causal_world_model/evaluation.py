"""
Offline evaluation utilities for the causal world model.
"""

from typing import Any, Dict, List

import pandas as pd

from .data_schema import Timeline


def doubly_robust_policy_value(
    behavior_log: pd.DataFrame,
    reward_col: str,
    action_col: str,
    propensities: pd.Series,
    q_values: pd.Series,
) -> float:
    """
    Compute a simple doubly robust estimate of policy value for offline evaluation.
    """
    dr_scores = propensities.apply(lambda e: 1.0 / max(e, 1e-6)) * (behavior_log[reward_col] - q_values)
    return float(q_values.mean() + dr_scores.mean())


def summarize_timeline_events(timeline: Timeline) -> Dict[str, int]:
    """
    Quick helper to count events by type for diagnostics.
    """
    counts: Dict[str, int] = {}
    for event in timeline:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return counts
