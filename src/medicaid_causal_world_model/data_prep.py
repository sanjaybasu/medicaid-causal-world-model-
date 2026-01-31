"""
Data preparation utilities for offline RL and causal modeling.

This module expects a stitched timeline with:
- member_id
- event_time (datetime)
- actions (categorical strings from taxonomy)
- outcomes (e.g., ADT event indicators)
- covariates and latent_z vectors

It converts events into fixed-step trajectories for offline RL (CQL/IQL/DT).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_schema import PatientEvent


@dataclass
class Trajectory:
    """Offline RL trajectory."""

    observations: np.ndarray  # shape (T, d_obs)
    actions: np.ndarray  # shape (T,)
    rewards: np.ndarray  # shape (T,)
    terminals: np.ndarray  # shape (T,)
    member_id: str


def encode_action(action: str, action_vocab: Dict[str, int]) -> int:
    return action_vocab.get(action, action_vocab.get("__unk__", -1))


def build_action_vocab(actions: Sequence[str]) -> Dict[str, int]:
    unique = sorted(set(actions))
    vocab = {a: i for i, a in enumerate(unique)}
    vocab["__unk__"] = len(vocab)
    return vocab


def timeline_to_trajectories(
    timeline: pd.DataFrame,
    action_col: str,
    reward_col: str,
    covariate_cols: Sequence[str],
    latent_col: Optional[str] = None,
    max_gap_days: int = 30,
) -> Tuple[List[Trajectory], Dict[str, int]]:
    """
    Convert a member-level event table into trajectories.

    Args:
        timeline: columns include member_id, event_time, action_col, reward_col, covariate_cols.
        action_col: name of the action column (categorical string).
        reward_col: name of the reward column (numeric; e.g., negative ADT indicator).
        covariate_cols: numeric covariate columns to include in state.
        latent_col: optional column containing array-like latent vectors.
        max_gap_days: start new trajectory if gap exceeds this.
    """
    trajectories: List[Trajectory] = []
    action_vocab = build_action_vocab(timeline[action_col].dropna().astype(str).tolist())

    for member_id, group in timeline.sort_values("event_time").groupby("member_id"):
        obs_list: List[np.ndarray] = []
        act_list: List[int] = []
        rew_list: List[float] = []
        term_list: List[int] = []

        prev_time = None
        for _, row in group.iterrows():
            ts = pd.to_datetime(row["event_time"])
            if prev_time is not None and (ts - prev_time).days > max_gap_days and obs_list:
                trajectories.append(
                    Trajectory(
                        observations=np.stack(obs_list, axis=0),
                        actions=np.array(act_list, dtype=np.int64),
                        rewards=np.array(rew_list, dtype=np.float32),
                        terminals=np.array(term_list, dtype=np.int64),
                        member_id=member_id,
                    )
                )
                obs_list, act_list, rew_list, term_list = [], [], [], []
            prev_time = ts

            covars = [row.get(c, 0.0) for c in covariate_cols]
            if latent_col and latent_col in row and row[latent_col] is not None:
                try:
                    covars.extend(list(row[latent_col]))
                except Exception:
                    pass
            obs_list.append(np.array(covars, dtype=np.float32))
            act_list.append(encode_action(str(row[action_col]), action_vocab))
            rew_list.append(float(row.get(reward_col, 0.0)))
            term_list.append(int(row.get("terminal", 0)))

        if obs_list:
            trajectories.append(
                Trajectory(
                    observations=np.stack(obs_list, axis=0),
                    actions=np.array(act_list, dtype=np.int64),
                    rewards=np.array(rew_list, dtype=np.float32),
                    terminals=np.array(term_list, dtype=np.int64),
                    member_id=member_id,
                )
            )

    return trajectories, action_vocab
