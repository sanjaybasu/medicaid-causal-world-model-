"""
Typed containers for stitched patient timelines.

These help keep the pipeline (timeline construction → perception → policy) consistent
across notebooks and package code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence


@dataclass
class PatientEvent:
    """One timestamped event from claims, encounters, or ADT."""

    member_id: str
    event_time: datetime
    event_type: str  # e.g., "encounter", "adt", "claim", "eligibility"
    action: Optional[str] = None  # e.g., "therapy", "pharmacy", "in_person", "pcp"
    outcome: Optional[str] = None  # e.g., "ed_visit", "ip_admit"
    features: Dict[str, float] = field(default_factory=dict)  # structured covariates
    text_features: Dict[str, float] = field(default_factory=dict)  # note-derived tags
    latent_z: Optional[Sequence[float]] = None  # DeepSCM confounder embedding
    censor_flag: bool = False  # for clone–censor–weight style emulation


@dataclass
class PatientState:
    """State vector at a given time for simulation or policy models."""

    member_id: str
    state_time: datetime
    covariates: Dict[str, float]
    latent_z: Optional[Sequence[float]] = None
    history_window: Optional[List[PatientEvent]] = None


@dataclass
class PolicyRecommendation:
    """Counterfactual policy output for a member-time."""

    member_id: str
    state_time: datetime
    recommended_action: str
    uplift: float  # negative = reduction in ADT risk
    uncertainty: float
    rationale: Dict[str, float] = field(default_factory=dict)  # top contributing features


Timeline = List[PatientEvent]
