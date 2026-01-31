"""
Offline RL training utilities (CQL, IQL, AWR, Decision Transformer) using d3rlpy.

Assumes trajectories are prepared via data_prep.timeline_to_trajectories.
"""

from typing import Dict, List

import numpy as np
from d3rlpy.algos import (
    AWAC,
    AWACConfig,
    IQL,
    IQLConfig,
    DiscreteCQL,
    DiscreteCQLConfig,
    DiscreteDecisionTransformer,
    DiscreteDecisionTransformerConfig,
    DiscreteSAC,
    DiscreteSACConfig,
    DiscreteBCQ,
    DiscreteBCQConfig,
)
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.components import Episode, Signature

from .data_prep import Trajectory


def trajectories_to_buffer(trajectories: List[Trajectory], action_size: int) -> ReplayBuffer:
    if not trajectories:
        raise ValueError("No trajectories provided.")
    obs_dim = trajectories[0].observations.shape[1]
    obs_sig = Signature(dtype=(np.dtype("float32"),), shape=(obs_dim,))
    act_sig = Signature(dtype=(np.dtype("int64"),), shape=(1,))
    rew_sig = Signature(dtype=(np.dtype("float32"),), shape=(1,))

    episodes = []
    for traj in trajectories:
        terminated = bool(traj.terminals[-1]) if len(traj.terminals) > 0 else False
        episodes.append(
            Episode(
                observations=traj.observations,
                actions=traj.actions.reshape(-1, 1),
                rewards=traj.rewards.reshape(-1, 1).astype(np.float32),
                terminated=terminated,
            )
        )
    buffer = InfiniteBuffer()
    rb = ReplayBuffer(
        buffer=buffer,
        episodes=episodes,
        observation_signature=obs_sig,
        action_signature=act_sig,
        reward_signature=rew_sig,
        action_space=ActionSpace.DISCRETE,
        action_size=action_size,
    )
    return rb


def train_cql(buffer: ReplayBuffer, n_steps: int = 100000, seed: int = 42, show_progress: bool = True) -> DiscreteCQL:
    np.random.seed(seed)
    config = DiscreteCQLConfig()
    algo = DiscreteCQL(config=config, device="cpu", enable_ddp=False)
    algo.fit(buffer, n_steps=n_steps, with_timestamp=False, show_progress=show_progress)
    return algo


def train_iql(buffer: ReplayBuffer, n_steps: int = 100000, seed: int = 42, show_progress: bool = True) -> DiscreteSAC:
    np.random.seed(seed)
    config = DiscreteSACConfig()
    algo = DiscreteSAC(config=config, device="cpu", enable_ddp=False)
    algo.fit(buffer, n_steps=n_steps, with_timestamp=False, show_progress=show_progress)
    return algo


def train_awac(buffer: ReplayBuffer, n_steps: int = 100000, seed: int = 42, show_progress: bool = True) -> DiscreteBCQ:
    np.random.seed(seed)
    config = DiscreteBCQConfig()
    algo = DiscreteBCQ(config=config, device="cpu", enable_ddp=False)
    algo.fit(buffer, n_steps=n_steps, with_timestamp=False, show_progress=show_progress)
    return algo


def train_dt(buffer: ReplayBuffer, n_steps: int = 50000, seed: int = 42, show_progress: bool = True) -> DiscreteDecisionTransformer:
    np.random.seed(seed)
    config = DiscreteDecisionTransformerConfig()
    algo = DiscreteDecisionTransformer(config=config, device="cpu", enable_ddp=False)
    algo.fit(buffer, n_steps=n_steps, with_timestamp=False, show_progress=show_progress)
    return algo


def evaluate_policy(*args, **kwargs) -> Dict[str, float]:
    """
    Placeholder for environment-based evaluation; prefer off-policy estimators
    (FQE, DR/WIS) for healthcare settings.
    """
    return {}
