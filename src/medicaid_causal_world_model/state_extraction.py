"""
Perception layer: converts unstructured encounters into structured states and latent
confounders suitable for causal modeling.

Key steps:
- Prompt/finetune an LLM to tag SDOH, trust, barriers, and staff actions from notes.
- Train a DeepSCM (VAE-backed) on embeddings to recover latent confounders Z.
- Attach Z to patient timelines for downstream CDT and SCM training.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .data_schema import PatientEvent, Timeline
from .taxonomy import bootstrap_tags


def extract_structured_tags(
    encounters: pd.DataFrame,
    prompt_template: str,
    llm_client: Any,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Run LLM tagging over encounter notes to produce structured SDOH/action labels.

    Args:
        encounters: columns include member_id, event_time, note_text.
        prompt_template: prompt string with placeholders for examples and note_text.
        llm_client: object with a .generate or .complete method.
        few_shot_examples: optional list of example dicts to inject into the prompt.

    Returns:
        encounters with added structured columns (e.g., housing_instability, trust_score, action_type).
    """
    raise NotImplementedError("Implement LLM tagging and parsing for encounter notes.")


def train_deep_scm(
    embeddings: pd.DataFrame,
    actions: pd.Series,
    outcomes: pd.Series,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train a VAE-based Deep Structural Causal Model to learn latent confounders Z.

    Args:
        embeddings: note or encounter embeddings aligned to actions/outcomes.
        actions: treatment labels aligned to embeddings.
        outcomes: ADT outcomes aligned to embeddings.
        config: optional model/training hyperparameters.

    Returns:
        Fitted DeepSCM model object with an .encode method.
    """
    raise NotImplementedError("Train DeepSCM to recover latent confounders from text.")


def attach_latent_z(
    timeline: Timeline,
    deep_scm_model: Any,
    embedding_lookup: Dict[str, List[float]],
    latent_key: str = "latent_z",
) -> Timeline:
    """
    Add latent confounder vectors to each encounter event within a timeline.

    Args:
        timeline: list of PatientEvent sorted by event_time.
        deep_scm_model: trained DeepSCM with .encode(embedding) -> latent vector.
        embedding_lookup: mapping from event ids or hashes to embeddings.
        latent_key: name of the PatientEvent attribute to store latents.

    Returns:
        Timeline with latent vectors attached to encounter events.
    """
    enriched: List[PatientEvent] = []
    for event in timeline:
        if event.event_type == "encounter":
            emb = embedding_lookup.get(f"{event.member_id}-{event.event_time.isoformat()}")
            if emb is not None:
                event.latent_z = deep_scm_model.encode(emb)  # type: ignore[attr-defined]
        enriched.append(event)
    return enriched


def bootstrap_structured_columns(encounters: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add high-precision heuristic tags to encounters to seed LLM prompts or models.

    Returns the input DataFrame with extra columns:
    - intervention_tags
    - sdoh_need_tags
    - clinical_support_tags
    - coordination_tags
    - behavioral_support_tags
    """
    tags: List[Dict[str, List[str]]] = []
    for _, row in encounters.iterrows():
        note = str(row.get(text_col, "") or "")
        tags.append(bootstrap_tags(note))
    encounters = encounters.copy()
    encounters["intervention_tags"] = [t["intervention"] for t in tags]
    encounters["sdoh_need_tags"] = [t["sdoh_need"] for t in tags]
    encounters["clinical_support_tags"] = [t["clinical_support"] for t in tags]
    encounters["coordination_tags"] = [t["coordination"] for t in tags]
    encounters["behavioral_support_tags"] = [t["behavioral_support"] for t in tags]
    return encounters
