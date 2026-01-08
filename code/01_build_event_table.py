"""
Step 1: Build Event Table
=========================
Construct patient trajectories from raw data sources.

This script:
1. Loads clinical notes, ADT feeds, and eligibility data
2. Links interventions to outcomes
3. Creates the RL-ready event table with state, action, reward structure

Inputs:
    - data/clinical_notes.parquet
    - data/adt_events.parquet
    - data/eligibility.parquet

Outputs:
    - outputs/rl_event_table.parquet
    - outputs/encounter_timeline.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_clinical_notes(path: str) -> pd.DataFrame:
    """Load clinical notes with encounter metadata."""
    print(f"Loading clinical notes from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} notes for {df['member_id'].nunique():,} members")
    return df

def load_adt_events(path: str) -> pd.DataFrame:
    """Load admission-discharge-transfer events."""
    print(f"Loading ADT events from {path}...")
    df = pd.read_parquet(path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    print(f"  Loaded {len(df):,} events")
    return df

def load_eligibility(path: str) -> pd.DataFrame:
    """Load eligibility and demographics."""
    print(f"Loading eligibility from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} members")
    return df

def compute_utilization_features(adt_df: pd.DataFrame, 
                                  reference_date: pd.Timestamp,
                                  member_id: str,
                                  lookback_days: int = 180) -> dict:
    """
    Compute utilization history features for a given member at a reference date.
    
    Returns:
        dict with hv_30d, hv_90d, hv_180d (hospital visit counts)
    """
    member_events = adt_df[adt_df['member_id'] == member_id].copy()
    member_events = member_events[member_events['event_date'] < reference_date]
    
    features = {}
    for days, key in [(30, 'hv_30d'), (90, 'hv_90d'), (180, 'hv_180d')]:
        cutoff = reference_date - pd.Timedelta(days=days)
        count = len(member_events[member_events['event_date'] >= cutoff])
        features[key] = count
    
    return features

def build_event_table(notes_df: pd.DataFrame,
                      adt_df: pd.DataFrame,
                      eligibility_df: pd.DataFrame,
                      config: dict) -> pd.DataFrame:
    """
    Build the main event table linking interventions to outcomes.
    
    Each row represents: (member_id, event_time, action, state, reward)
    """
    print("Building event table...")
    
    lookback = config['features']['lookback_days']
    outcome_window = config['features']['outcome_window_days']
    
    # Merge notes with eligibility
    df = notes_df.merge(eligibility_df, on='member_id', how='left')
    
    # Sort by member and time
    df = df.sort_values(['member_id', 'event_time'])
    
    # Compute outcome: any acute event in next 30 days
    print("  Computing outcomes (acute events in next 30 days)...")
    outcomes = []
    for idx, row in df.iterrows():
        member_events = adt_df[adt_df['member_id'] == row['member_id']]
        start = row['event_time']
        end = start + pd.Timedelta(days=outcome_window)
        acute_count = len(member_events[
            (member_events['event_date'] >= start) & 
            (member_events['event_date'] < end)
        ])
        outcomes.append(-1 if acute_count > 0 else 0)  # Reward: -1 for event
    
    df['reward'] = outcomes
    
    # Compute utilization features
    print("  Computing utilization history features...")
    hv_features = df.apply(
        lambda row: compute_utilization_features(
            adt_df, row['event_time'], row['member_id'], lookback
        ), axis=1
    )
    hv_df = pd.DataFrame(hv_features.tolist())
    df = pd.concat([df.reset_index(drop=True), hv_df], axis=1)
    
    # Add terminal flag (last event per member)
    df['terminal'] = df.groupby('member_id').cumcount(ascending=False) == 0
    
    print(f"  Built event table: {len(df):,} rows, {df['member_id'].nunique():,} members")
    
    return df

def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 1: Build Event Table")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Create output directory
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    notes_df = load_clinical_notes(config['data']['clinical_notes'])
    adt_df = load_adt_events(config['data']['adt_feeds'])
    eligibility_df = load_eligibility(config['data']['eligibility'])
    
    # Build event table
    event_table = build_event_table(notes_df, adt_df, eligibility_df, config)
    
    # Save outputs
    output_path = output_dir / "rl_event_table.parquet"
    event_table.to_parquet(output_path, index=False)
    print(f"\nSaved event table to {output_path}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total events: {len(event_table):,}")
    print(f"Unique members: {event_table['member_id'].nunique():,}")
    print(f"Acute event rate: {(event_table['reward'] == -1).mean():.1%}")
    print(f"Mean hv_90d: {event_table['hv_90d'].mean():.2f}")

if __name__ == "__main__":
    main()
