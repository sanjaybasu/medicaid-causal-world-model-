"""
Step 5: Compute Treatment Effects
==================================
Doubly robust estimation of conditional average treatment effects (CATE).

This script:
1. Stratifies patients by receptivity state
2. Computes CATE within each stratum using doubly robust estimation
3. Calculates Numbers Needed to Treat (NNT)
4. Performs sensitivity analysis (E-values)

Inputs:
    - outputs/rl_event_table_enriched.parquet

Outputs:
    - outputs/treatment_effects.json
    - outputs/causal_dr_results.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_event_table(path: str) -> pd.DataFrame:
    """Load the enriched event table."""
    print(f"Loading event table from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows")
    return df

def define_receptivity_states(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assign receptivity state to each observation.
    
    States:
        - high: >=2 ED visits in 90 days
        - moderate: 1 ED visit in 90 days
        - low: 0 ED visits in 180 days
    """
    print("Defining receptivity states...")
    
    recept_config = config['causal']['receptivity']
    
    def assign_state(row):
        if row['hv_90d'] >= recept_config['high']['ed_visits_90d']:
            return 'high'
        elif row['hv_90d'] >= recept_config['moderate']['ed_visits_90d']:
            return 'moderate'
        else:
            return 'low'
    
    df['receptivity_state'] = df.apply(assign_state, axis=1)
    
    # Print distribution
    print("  State distribution:")
    for state, count in df['receptivity_state'].value_counts().items():
        print(f"    {state}: {count:,} ({count/len(df):.1%})")
    
    return df

def doubly_robust_ate(df: pd.DataFrame,
                      treatment_col: str,
                      outcome_col: str,
                      covariate_cols: list,
                      n_folds: int = 5) -> dict:
    """
    Compute Average Treatment Effect using doubly robust estimation.
    
    Uses cross-fitting to avoid overfitting bias.
    
    Args:
        df: DataFrame with treatment, outcome, and covariates
        treatment_col: Name of binary treatment column
        outcome_col: Name of binary outcome column
        covariate_cols: List of covariate column names
        n_folds: Number of folds for cross-fitting
    
    Returns:
        dict with ate, se, ci_lower, ci_upper
    """
    X = df[covariate_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    n = len(df)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Cross-fitted propensity scores
    propensity_model = LogisticRegression(max_iter=1000, solver='lbfgs')
    e_hat = cross_val_predict(propensity_model, X, T, cv=cv, method='predict_proba')[:, 1]
    e_hat = np.clip(e_hat, 0.01, 0.99)  # Clip for stability
    
    # Cross-fitted outcome models
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    
    for train_idx, test_idx in cv.split(X, T):
        # Fit outcome model on treated
        treated_idx = train_idx[T[train_idx] == 1]
        if len(treated_idx) > 10:
            model_1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            model_1.fit(X[treated_idx], Y[treated_idx])
            mu1_hat[test_idx] = model_1.predict_proba(X[test_idx])[:, 1]
        
        # Fit outcome model on control
        control_idx = train_idx[T[train_idx] == 0]
        if len(control_idx) > 10:
            model_0 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            model_0.fit(X[control_idx], Y[control_idx])
            mu0_hat[test_idx] = model_0.predict_proba(X[test_idx])[:, 1]
    
    # Doubly robust estimator
    dr_scores = (
        mu1_hat - mu0_hat +
        T * (Y - mu1_hat) / e_hat -
        (1 - T) * (Y - mu0_hat) / (1 - e_hat)
    )
    
    ate = np.mean(dr_scores)
    se = np.std(dr_scores) / np.sqrt(n)
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    
    return {
        "ate": float(ate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": int(n)
    }

def compute_nnt(ate: float, ci_lower: float, ci_upper: float) -> dict:
    """
    Compute Number Needed to Treat and its confidence interval.
    
    NNT = 1 / ARR (absolute risk reduction)
    For NNT CI, invert the ATE CI bounds (swap order since NNT is inverse)
    """
    if ate <= 0:
        return {"nnt": float('inf'), "nnt_ci_lower": float('inf'), "nnt_ci_upper": float('inf')}
    
    nnt = 1 / ate
    nnt_ci_lower = 1 / ci_upper if ci_upper > 0 else float('inf')
    nnt_ci_upper = 1 / ci_lower if ci_lower > 0 else float('inf')
    
    return {
        "nnt": float(round(nnt, 1)),
        "nnt_ci_lower": float(round(nnt_ci_lower, 1)),
        "nnt_ci_upper": float(round(nnt_ci_upper, 1))
    }

def compute_e_value(ate: float, ci_lower: float) -> float:
    """
    Compute E-value for sensitivity to unmeasured confounding.
    
    E-value = RR + sqrt(RR * (RR - 1))
    where RR is approximate risk ratio from ATE
    """
    if ate <= 0 or ci_lower <= 0:
        return float('inf')
    
    # Convert ATE to approximate RR (assuming baseline risk of 0.5)
    baseline_risk = 0.5
    rr = (baseline_risk - ate) / baseline_risk
    if rr <= 1:
        return 1.0
    
    e_value = rr + np.sqrt(rr * (rr - 1))
    return round(e_value, 2)

def main():
    """Main entry point."""
    print("=" * 60)
    print("Step 5: Compute Treatment Effects")
    print("=" * 60)
    
    # Load config
    config = load_config()
    output_dir = Path(config['data']['output_dir'])
    
    # Load data
    df = load_event_table(output_dir / "rl_event_table_enriched.parquet")
    
    # Define receptivity states
    df = define_receptivity_states(df, config)
    
    # Define treatment (phone call intervention)
    df['treated'] = df['action'].apply(lambda x: 1 if 'phone' in str(x).lower() else 0)
    
    # Define outcome (acute event = reward of -1)
    df['acute_event'] = (df['reward'] == -1).astype(int)
    
    # Define covariates
    covariate_cols = ['age_years', 'hv_30d', 'hv_90d', 'hv_180d']
    covariate_cols = [c for c in covariate_cols if c in df.columns]
    
    # Compute CATE by receptivity state
    print("\n--- Computing CATEs by Receptivity State ---")
    results = {}
    
    for state in ['high', 'moderate', 'low']:
        state_df = df[df['receptivity_state'] == state]
        print(f"\n{state.upper()} receptivity (n={len(state_df):,}):")
        
        if len(state_df) < 100:
            print("  Insufficient sample size")
            continue
        
        # Compute ATE
        ate_result = doubly_robust_ate(
            state_df, 'treated', 'acute_event', covariate_cols,
            n_folds=config['causal']['n_folds']
        )
        
        # Compute NNT
        nnt_result = compute_nnt(
            ate_result['ate'], ate_result['ci_lower'], ate_result['ci_upper']
        )
        
        # Compute E-value
        e_value = compute_e_value(ate_result['ate'], ate_result['ci_lower'])
        
        results[state] = {
            **ate_result,
            **nnt_result,
            "e_value": e_value
        }
        
        print(f"  ATE: {ate_result['ate']:.4f} (95% CI: {ate_result['ci_lower']:.4f} to {ate_result['ci_upper']:.4f})")
        print(f"  NNT: {nnt_result['nnt']} (95% CI: {nnt_result['nnt_ci_lower']} to {nnt_result['nnt_ci_upper']})")
        print(f"  E-value: {e_value}")
    
    # Save results
    output_path = output_dir / "treatment_effects.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved treatment effects to {output_path}")
    
    print("\n--- Summary ---")
    if 'high' in results and 'low' in results:
        ratio = results['low']['nnt'] / results['high']['nnt']
        print(f"NNT ratio (low/high receptivity): {ratio:.0f}x")

if __name__ == "__main__":
    main()
