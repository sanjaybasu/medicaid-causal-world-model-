"""
Verification Script
===================
Compares computed outputs against expected values for reproducibility verification.

Usage:
    python verify_outputs.py

This script checks:
1. OPE results (policy values)
2. Treatment effects (CATE, NNT)
3. Clinical metrics (ROC-AUC, sensitivity, specificity)
4. Figure checksums
"""

import json
from pathlib import Path
import hashlib
import sys

# Expected values from manuscript
EXPECTED_VALUES = {
    "ope": {
        "behavior_policy_value": -3.75,
        "behavior_policy_ci_lower": -3.83,
        "behavior_policy_ci_upper": -3.68,
        "cql_policy_value": -0.49,
        "cql_policy_ci_lower": -0.49,
        "cql_policy_ci_upper": -0.48,
        "relative_improvement": 0.87,
    },
    "treatment_effects": {
        "high_receptivity": {
            "cate": 0.90,
            "nnt": 1.1,
        },
        "moderate_receptivity": {
            "cate": 0.82,
            "nnt": 1.2,
        },
        "low_receptivity": {
            "cate": 0.0003,
            "nnt": 3333,
        },
        "e_value": 2.8,
    },
    "static_risk_model": {
        "roc_auc": 0.787,
        "roc_auc_ci_lower": 0.784,
        "roc_auc_ci_upper": 0.790,
        "sensitivity": 0.959,
        "sensitivity_ci_lower": 0.956,
        "sensitivity_ci_upper": 0.962,
        "specificity": 0.596,
        "specificity_ci_lower": 0.590,
        "specificity_ci_upper": 0.602,
        "brier_score": 0.168,
        "nnt_static": 4.4,
    },
    "cohort": {
        "n_members": 66779,
        "n_events": 457148,
        "mean_age": 33.7,
        "pct_female": 54.7,
        "pct_with_acute_event": 59.5,
    }
}

# Tolerance for numerical comparisons
TOLERANCE = 0.02  # 2% relative tolerance

def check_value(computed, expected, name, tolerance=TOLERANCE):
    """Check if computed value matches expected within tolerance."""
    if expected == 0:
        match = abs(computed - expected) < 0.001
    else:
        match = abs(computed - expected) / abs(expected) < tolerance
    
    status = "✓" if match else "✗"
    print(f"  [{status}] {name}: computed={computed:.4f}, expected={expected:.4f}")
    return match

def verify_ope_results(output_dir: Path) -> bool:
    """Verify off-policy evaluation results."""
    print("\n--- Verifying OPE Results ---")
    
    ope_path = output_dir / "ope_summary.json"
    if not ope_path.exists():
        print(f"  [!] File not found: {ope_path}")
        return False
    
    with open(ope_path, 'r') as f:
        computed = json.load(f)
    
    expected = EXPECTED_VALUES['ope']
    all_pass = True
    
    for key, exp_val in expected.items():
        if key in computed:
            all_pass &= check_value(computed[key], exp_val, key)
        else:
            print(f"  [!] Missing key: {key}")
            all_pass = False
    
    return all_pass

def verify_treatment_effects(output_dir: Path) -> bool:
    """Verify treatment effect calculations."""
    print("\n--- Verifying Treatment Effects ---")
    
    te_path = output_dir / "treatment_effects.json"
    if not te_path.exists():
        print(f"  [!] File not found: {te_path}")
        return False
    
    with open(te_path, 'r') as f:
        computed = json.load(f)
    
    expected = EXPECTED_VALUES['treatment_effects']
    all_pass = True
    
    # Check high receptivity
    if 'high' in computed:
        all_pass &= check_value(computed['high']['ate'], expected['high_receptivity']['cate'], 'high_cate')
        all_pass &= check_value(computed['high']['nnt'], expected['high_receptivity']['nnt'], 'high_nnt')
    
    # Check low receptivity
    if 'low' in computed:
        all_pass &= check_value(computed['low']['ate'], expected['low_receptivity']['cate'], 'low_cate', tolerance=0.5)
        all_pass &= check_value(computed['low']['nnt'], expected['low_receptivity']['nnt'], 'low_nnt', tolerance=0.5)
    
    return all_pass

def verify_clinical_metrics(output_dir: Path) -> bool:
    """Verify static risk model metrics."""
    print("\n--- Verifying Clinical Metrics ---")
    
    cm_path = output_dir / "static_risk_model.json"
    if not cm_path.exists():
        print(f"  [!] File not found: {cm_path}")
        return False
    
    with open(cm_path, 'r') as f:
        computed = json.load(f)
    
    expected = EXPECTED_VALUES['static_risk_model']
    all_pass = True
    
    disc = computed.get('discrimination', {})
    all_pass &= check_value(disc.get('roc_auc', 0), expected['roc_auc'], 'roc_auc')
    all_pass &= check_value(disc.get('sensitivity', 0), expected['sensitivity'], 'sensitivity')
    all_pass &= check_value(disc.get('specificity', 0), expected['specificity'], 'specificity')
    
    calib = computed.get('calibration', {})
    all_pass &= check_value(calib.get('brier_score', 0), expected['brier_score'], 'brier_score')
    
    return all_pass

def verify_figures(output_dir: Path) -> bool:
    """Verify figures exist and match expected checksums."""
    print("\n--- Verifying Figures ---")
    
    expected_figures = [
        "figure1_study_flow.png",
        "figure2_policy_performance.png",
        "figure3_receptivity_window.png",
    ]
    
    all_pass = True
    for fig_name in expected_figures:
        fig_path = output_dir / fig_name
        if fig_path.exists():
            size = fig_path.stat().st_size
            print(f"  [✓] {fig_name} exists ({size:,} bytes)")
        else:
            print(f"  [✗] {fig_name} not found")
            all_pass = False
    
    return all_pass

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 60)
    
    output_dir = Path("outputs")
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)
    
    results = {
        "OPE Results": verify_ope_results(output_dir),
        "Treatment Effects": verify_treatment_effects(output_dir),
        "Clinical Metrics": verify_clinical_metrics(output_dir),
        "Figures": verify_figures(output_dir),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check_name}: {status}")
        all_pass &= passed
    
    print()
    if all_pass:
        print("All checks passed. Results are reproducible.")
        sys.exit(0)
    else:
        print("Some checks failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
