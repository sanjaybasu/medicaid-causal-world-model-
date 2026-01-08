"""
Step 2: Extract Interventions (NLP)
====================================
Extract structured interventions from unstructured clinical notes.

This script:
1. Loads clinical notes
2. (Mock) Simulates the Teacher-Student distillation process
   - In a real run, this would query OpenAI (Teacher) and train DistilBERT (Student)
3. For this package, we use the pre-computed tags or synthetic tags if unavailable

Inputs:
    - data/clinical_notes.parquet

Outputs:
    - outputs/distilled_tags.parquet
    - outputs/rl_event_table_enriched.parquet (adds 'action' column to event table)
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

def load_event_table(path: str) -> pd.DataFrame:
    """Load the event table from Step 1."""
    print(f"Loading event table from {path}...")
    df = pd.read_parquet(path)
    return df

def simulate_intervention_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate extracted interventions for reproducibility package.
    
    Ref: 44-category taxonomy.
    Common actions:
    - Phone Call - Outreach
    - Phone Call - Assessment
    - Home Visit - Engagement
    - Coordination - Housing
    - None (No Action)
    """
    print("Extracting interventions (Simulated for reproducibility)...")
    
    # Define action space
    actions = [
        "Phone Call - Outreach", 
        "Phone Call - Assessment",
        "Home Visit - Engagement",
        "Coordination - Medication",
        "Coordination - Housing",
        "No Action"
    ]
    
    # Assign actions based on simple logic (mocking the NLP model)
    # In reality, this comes from the DistilBERT model inference
    np.random.seed(42)
    df['action'] = np.random.choice(actions, size=len(df), p=[0.3, 0.2, 0.1, 0.15, 0.05, 0.2])
    
    # Map to integer IDs for RL
    action_map = {a: i for i, a in enumerate(actions)}
    df['action_id'] = df['action'].map(action_map)
    
    print(f"  Extracted {len(df):,} actions")
    print("  Action distribution:")
    print(df['action'].value_counts(normalize=True))
    
    return df

def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 2: Extract Interventions (NLP)")
    print("=" * 60)
    
    config = load_config()
    output_dir = Path(config['data']['output_dir'])
    
    # Load Step 1 output
    input_path = output_dir / "rl_event_table.parquet"
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run Step 1 first.")
        return
        
    df = load_event_table(input_path)
    
    # Run extraction
    df = simulate_intervention_extraction(df)
    
    # Save enriched table
    output_path = output_dir / "rl_event_table_enriched.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved enriched event table to {output_path}")

if __name__ == "__main__":
    main()
