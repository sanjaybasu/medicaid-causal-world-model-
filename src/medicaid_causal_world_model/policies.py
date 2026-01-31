"""
Policy utilities for mapping model outputs to CHW-facing recommendations.
"""

from typing import List, Sequence

from .data_schema import PolicyRecommendation


def rank_actions(
    recommendations: List[PolicyRecommendation],
    budget: int,
) -> List[PolicyRecommendation]:
    """
    Rank action recommendations by uplift subject to capacity/budget.

    Args:
        recommendations: list of per-member action candidates.
        budget: number of actions that can be executed in the time window.
    """
    ranked = sorted(recommendations, key=lambda rec: rec.uplift)
    return ranked[:budget]


def apply_fairness_guardrails(
    recommendations: List[PolicyRecommendation],
    max_gap: float = 0.05,
) -> List[PolicyRecommendation]:
    """
    Placeholder for fairness-aware re-ranking (e.g., equal opportunity caps).

    Args:
        recommendations: action list.
        max_gap: allowable disparity in uplift thresholds across protected groups.
    """
    # TODO: implement fairness-aware re-ranking once demographic bins are available.
    return recommendations
