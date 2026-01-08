"""
Step 3: Train Offline RL Models
================================
Train CQL, IQL, and BCQ policies using d3rlpy.

This script:
1. Loads the enriched event table with intervention labels
2. Converts to d3rlpy MDPDataset format
3. Trains three offline RL algorithms
4. Saves trained models

Inputs:
    - outputs/rl_event_table_enriched.parquet

Outputs:
    - outputs/cql.d3
    - outputs/iql.d3
    - outputs/bcq.d3
    - outputs/rl_training_log.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import torch
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig, IQLConfig, BCQConfig
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seeds(config: dict):
    """Set random seeds for reproducibility."""
    np.random.seed(config['numpy_seed'])
    torch.manual_seed(config['torch_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['torch_seed'])

def load_event_table(path: str) -> pd.DataFrame:
    """Load the enriched event table."""
    print(f"Loading event table from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows, {df['member_id'].nunique():,} members")
    return df

def create_mdp_dataset(df: pd.DataFrame, 
                       state_cols: list,
                       action_col: str = 'action') -> MDPDataset:
    """
    Convert DataFrame to d3rlpy MDPDataset.
    
    Args:
        df: Event table with state, action, reward, terminal columns
        state_cols: List of columns to use as state features
        action_col: Column name for actions (discrete)
    
    Returns:
        MDPDataset ready for training
    """
    print("Creating MDP dataset...")
    
    # Extract arrays
    observations = df[state_cols].values.astype(np.float32)
    actions = df[action_col].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)
    terminals = df['terminal'].values.astype(bool)
    
    # Create dataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )
    
    print(f"  Observations shape: {observations.shape}")
    print(f"  Unique actions: {len(np.unique(actions))}")
    print(f"  Reward distribution: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
    print(f"  Terminal rate: {terminals.mean():.3f}")
    
    return dataset

def train_cql(dataset: MDPDataset, config: dict, output_dir: Path) -> dict:
    """Train Conservative Q-Learning model."""
    print("\n--- Training CQL ---")
    
    rl_config = config['rl']
    cql_config = rl_config.get('cql', {})
    
    # Configure CQL
    cql = CQLConfig(
        learning_rate=rl_config['learning_rate'],
        batch_size=rl_config['batch_size'],
        gamma=rl_config['gamma'],
        alpha=cql_config.get('alpha', 1.0),
    ).create(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    cql.fit(
        dataset,
        n_steps=rl_config['n_steps'],
        show_progress=True,
    )
    
    # Save
    model_path = output_dir / "cql.d3"
    cql.save(str(model_path))
    print(f"  Saved CQL model to {model_path}")
    
    return {"algorithm": "CQL", "n_steps": rl_config['n_steps']}

def train_iql(dataset: MDPDataset, config: dict, output_dir: Path) -> dict:
    """Train Implicit Q-Learning model."""
    print("\n--- Training IQL ---")
    
    rl_config = config['rl']
    iql_config = rl_config.get('iql', {})
    
    # Configure IQL
    iql = IQLConfig(
        learning_rate=rl_config['learning_rate'],
        batch_size=rl_config['batch_size'],
        gamma=rl_config['gamma'],
        expectile=iql_config.get('expectile', 0.7),
        temperature=iql_config.get('temperature', 3.0),
    ).create(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    iql.fit(
        dataset,
        n_steps=rl_config['n_steps'],
        show_progress=True,
    )
    
    # Save
    model_path = output_dir / "iql.d3"
    iql.save(str(model_path))
    print(f"  Saved IQL model to {model_path}")
    
    return {"algorithm": "IQL", "n_steps": rl_config['n_steps']}

def train_bcq(dataset: MDPDataset, config: dict, output_dir: Path) -> dict:
    """Train Batch-Constrained Q-Learning model."""
    print("\n--- Training BCQ ---")
    
    rl_config = config['rl']
    bcq_config = rl_config.get('bcq', {})
    
    # Configure BCQ (discrete version)
    bcq = BCQConfig(
        learning_rate=rl_config['learning_rate'],
        batch_size=rl_config['batch_size'],
        gamma=rl_config['gamma'],
    ).create(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    bcq.fit(
        dataset,
        n_steps=rl_config['n_steps'],
        show_progress=True,
    )
    
    # Save
    model_path = output_dir / "bcq.d3"
    bcq.save(str(model_path))
    print(f"  Saved BCQ model to {model_path}")
    
    return {"algorithm": "BCQ", "n_steps": rl_config['n_steps']}

def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 3: Train Offline RL Models")
    print("=" * 60)
    
    # Load config and set seeds
    config = load_config()
    set_seeds(config)
    
    output_dir = Path(config['data']['output_dir'])
    
    # Load data
    event_table = load_event_table(output_dir / "rl_event_table_enriched.parquet")
    
    # Define state columns
    state_cols = [
        'age_years', 'hv_30d', 'hv_90d', 'hv_180d',
        # Add one-hot encoded features
    ] + [c for c in event_table.columns if c.startswith('contactType_')]
    
    # Filter to available columns
    state_cols = [c for c in state_cols if c in event_table.columns]
    print(f"Using {len(state_cols)} state features")
    
    # Create dataset
    dataset = create_mdp_dataset(event_table, state_cols)
    
    # Train models
    training_log = []
    
    for algorithm in config['rl']['algorithms']:
        if algorithm == 'cql':
            log = train_cql(dataset, config, output_dir)
        elif algorithm == 'iql':
            log = train_iql(dataset, config, output_dir)
        elif algorithm == 'bcq':
            log = train_bcq(dataset, config, output_dir)
        else:
            print(f"Unknown algorithm: {algorithm}")
            continue
        training_log.append(log)
    
    # Save training log
    log_path = output_dir / "rl_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\nSaved training log to {log_path}")
    
    print("\n--- Training Complete ---")
    print(f"Trained {len(training_log)} models")

if __name__ == "__main__":
    main()
