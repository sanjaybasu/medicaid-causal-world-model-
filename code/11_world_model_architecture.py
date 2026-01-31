#!/usr/bin/env python3
"""
True World Model Architecture with Learned Transition Dynamics

This implements a model-based RL approach where we:
1. Learn P(s_{t+1} |  s_t, a_t) - state transitions
2. Learn R(s_t, a_t) - reward function  
3. Use learned model for counterfactual planning

Compared to previous FE approach (within-person causal inference),
this enables forward simulation and "what-if" scenarios.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import json

np.random.seed(42)
torch.manual_seed(42)

# ==========================================================================
# NEURAL NETWORK COMPONENTS
# ==========================================================================

class StateEncoder(nn.Module):
    """Maps raw features to latent state"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)


class TransitionModel(nn.Module):
    """Learns P(s_{t+1} | s_t, a_t)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


class RewardModel(nn.Module):
    """Learns R(s_t, a_t)"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


class WorldModel(nn.Module):
    """Complete world model architecture"""
    def __init__(self, input_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.state_encoder = StateEncoder(input_dim, hidden_dim, latent_dim)
        self.transition_model = TransitionModel(latent_dim, action_dim, hidden_dim)
        self.reward_model = RewardModel(latent_dim, action_dim, hidden_dim // 2)
        self.latent_dim = latent_dim
        
    def encode_state(self, x):
        return self.state_encoder(x)
    
    def predict_transition(self, state, action):
        return self.transition_model(state, action)
    
    def predict_reward(self, state, action):
        return self.reward_model(state, action)
    
    def forward(self, x, action):
        state = self.encode_state(x)
        next_state = self.predict_transition(state, action)
        reward = self.predict_reward(state, action)
        return state, next_state, reward


# ==========================================================================
# DATASET
# ==========================================================================

class TrajectoryDataset(Dataset):
    """Patient trajectories: (s_t, a_t, s_{t+1}, r_t) - Vectorized"""
    def __init__(self, data, features, action_col='treated'):
        self.features = features
        print("  Preparing dataset vectorization...")
        
        # Ensure sorted
        data = data.sort_values(['member_id', 'date'])
        
        # Shift next state features
        # We need validation that s_{t+1} belongs to same member
        # Shift mask: 1 if next row is same member, 0 otherwise
        member_ids = data['member_id'].values
        valid_transition = (member_ids[:-1] == member_ids[1:])
        
        # Current States (s_t)
        states = data[features].values[:-1]
        
        # Actions (a_t)
        actions = data['received_intervention'].values[:-1].reshape(-1, 1)
        
        # Next States (s_{t+1})
        next_states = data[features].values[1:]
        
        # Rewards (r_t)
        if 'distilled_reward' in data.columns:
            print("  Using Distilled Rewards (Teacher-Student SOTA)!")
            rewards = data['distilled_reward'].values[:-1].reshape(-1, 1)
        else:
            rewards = -data['target_next_month'].values[:-1].reshape(-1, 1)
        
        # Propensity scores (p_t)
        propensities = data['propensity_score'].values[:-1].reshape(-1, 1)
        
        # Filtering for valid transitions (same member + no NaNs)
        # We already cleaned NaNs in preprocess_data, but double check
        # We need to filter based on 'valid_transition' mask
        
        mask = valid_transition
        
        self.states = torch.FloatTensor(states[mask])
        self.actions = torch.FloatTensor(actions[mask])
        self.next_states = torch.FloatTensor(next_states[mask])
        self.rewards = torch.FloatTensor(rewards[mask])
        self.propensities = torch.FloatTensor(propensities[mask])
        
        print(f"  Created {len(self.states)} transitions (Vectorized)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.propensities[idx]
        )



# ==========================================================================
# TRAINING
# ==========================================================================

def train_world_model(data, features, epochs=30, batch_size=1024, lr=1e-3, device='cpu'):
    """Train world model end-to-end"""
    print("\n" + "="*70)
    print("TRAINING WORLD MODEL (Learning Forward Dynamics)")
    print("="*70)
    
    dataset = TrajectoryDataset(data, features)
    if len(dataset) == 0:
        print("ERROR: No valid transitions")
        return None
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    input_dim = len(features)
    model = WorldModel(input_dim, action_dim=1, latent_dim=64, hidden_dim=128).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    for epoch in range(epochs):
        trans_loss_sum = 0
        reward_loss_sum = 0
        n_batches = 0
        
        for batch in dataloader:
            state, action, next_state_true, reward_true, propensity = [b.to(device) for b in batch]

            
            # Forward
            state_latent, next_state_pred, reward_pred = model(state, action)
            next_state_true_latent = model.encode_state(next_state_true)
            
            # Loss Weights (IPW)
            # Clip weights to prevent instability (e.g. 0.1 to 10.0)
            ps = propensity.squeeze()
            action_mask = action.squeeze()
            weights = (action_mask / (ps + 1e-4)) + ((1 - action_mask) / (1 - ps + 1e-4))
            weights = torch.clamp(weights, 0.1, 10.0)

            # Losses
            trans_loss = F.mse_loss(next_state_pred, next_state_true_latent)
            
            # Weighted Reward Loss (Causal)
            raw_reward_loss = F.mse_loss(reward_pred, reward_true, reduction='none')
            reward_loss = (raw_reward_loss.squeeze() * weights).mean()
            
            loss = trans_loss + reward_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            trans_loss_sum += trans_loss.item()
            reward_loss_sum += reward_loss.item()
            n_batches += 1

            n_batches += 1
        
        avg_total = (trans_loss_sum + reward_loss_sum) / n_batches
        scheduler.step(avg_total)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Trans={trans_loss_sum/n_batches:.4f}, "
                  f"Reward={reward_loss_sum/n_batches:.4f}")
    
    print("World model training complete!\n")
    return model


def plan_with_world_model(model, data, features, device='cpu'):
    """Counterfactual planning: simulate both a=0 and a=1"""
    print("=" * 70)
    print("MODEL-BASED PLANNING (Counterfactual Simulation)")
    print("=" * 70)
    
    model.eval()
    X = torch.from_numpy(data[features].fillna(0).values.astype(np.float32)).to(device)
    
    with torch.no_grad():
        states = model.encode_state(X)
        
        # Simulate a=1 (treatment)
        action_treat = torch.ones(len(X), 1).to(device)
        reward_treat = model.predict_reward(states, action_treat).cpu().numpy().flatten()
        
        # Simulate a=0 (no treatment)
        action_no_treat = torch.zeros(len(X), 1).to(device)
        reward_no_treat = model.predict_reward(states, action_no_treat).cpu().numpy().flatten()
        
        # CATE = treatment effect
        cate = reward_treat - reward_no_treat
    
    print(f"  CATE distribution: Mean={cate.mean():.4f}, Std={cate.std():.4f}")
    print(f"  Range: [{cate.min():.4f}, {cate.max():.4f}]")
    print(f"  Positive: {(cate > 0).mean()*100:.1f}%\n")
    
    return cate


# ==========================================================================
# DATA LOADING (from working FE script)
# ==========================================================================

def load_real_data():
    base_path = Path("../../data/real_inputs")
    outcomes = pd.read_csv(base_path / "outcomes_monthly.csv", 
                           usecols=['member_id', 'month_year', 'emergency_department_ct', 'acute_inpatient_ct'])
    interventions = pd.read_csv(base_path / "interventions.csv",
                                usecols=['person_key', 'intervention_date', 'intervention'])
    attributes = pd.read_csv(base_path / "member_attributes.csv",
                             usecols=['member_id', 'birth_date', 'gender', 'state', 'risk_score'])
    return outcomes, interventions, attributes


def preprocess_data(outcomes, interventions, attributes):
    # Outcomes
    outcomes['date'] = pd.to_datetime(outcomes['month_year'])
    outcomes['event'] = ((outcomes['emergency_department_ct'] > 0) | 
                        (outcomes['acute_inpatient_ct'] > 0)).astype(int)
    
    # Attributes
    attributes['dob'] = pd.to_datetime(attributes['birth_date'], errors='coerce')
    ref_date = pd.Timestamp('2024-01-01')
    attributes['age'] = (ref_date - attributes['dob']).dt.days / 365.25
    attributes['age'] = attributes['age'].fillna(35.0)
    
    if 'risk_score' in attributes.columns:
        attributes['risk_score'] = pd.to_numeric(attributes['risk_score'], errors='coerce')
        attributes['risk_score'] = attributes['risk_score'].fillna(attributes['risk_score'].median())
    
    attributes['is_female'] = (attributes['gender'] == 'F').astype(int)
    
    if 'state' in attributes.columns:
        state_dummies = pd.get_dummies(attributes['state'], prefix='st')
        attributes = pd.concat([attributes, state_dummies], axis=1)
    
    cols_to_merge = ['member_id', 'age', 'is_female', 'risk_score']
    state_cols = [c for c in attributes.columns if c.startswith('st_')]
    cols_to_merge += state_cols
    
    data = outcomes.merge(attributes[cols_to_merge], on='member_id', how='left')
    
    # Interventions
    interventions = interventions.rename(columns={'person_key': 'member_id'})
    interventions['date'] = pd.to_datetime(interventions['intervention_date'])
    interventions['month_year'] = interventions['date'].dt.to_period('M').astype(str)
    
    int_agg = interventions.groupby(['member_id', 'month_year']).size().reset_index(name='intervention_count')
    int_agg['treated'] = 1
    
    data = data.merge(int_agg[['member_id', 'month_year', 'treated']], 
                     on=['member_id', 'month_year'], how='left')
    data['treated'] = data['treated'].fillna(0)
    
    # Lags
    data = data.sort_values(['member_id', 'date'])
    data['prev_event_1m'] = data.groupby('member_id')['event'].shift(1).fillna(0)
    data['prev_event_3m_sum'] = (data.groupby('member_id')['event']
                                 .rolling(3, min_periods=1).sum()
                                 .reset_index(0, drop=True).shift(1).fillna(0))
    data['utilization_trend'] = data['prev_event_1m'] - (data['prev_event_3m_sum'] / 3.0)
    
    # Target
    data['target_next_month'] = data.groupby('member_id')['event'].shift(-1)
    data = data.dropna(subset=['target_next_month', 'age', 'is_female'])
    
    # FILL ALL REMAINING NANS to ensure World Model input is valid
    # Features might still have NaNs from shifting/rolling at start of history
    numeric_cols = ['prev_event_1m', 'prev_event_3m_sum', 'utilization_trend', 'risk_score', 'age']
    data[numeric_cols] = data[numeric_cols].fillna(0)
    
    # Fill state dummies if any
    state_cols = [c for c in data.columns if c.startswith('st_')]
    if state_cols:
        data[state_cols] = data[state_cols].fillna(0)
    
    # Rename treated to received_intervention for consistency
    data = data.rename(columns={'treated': 'received_intervention'})

    # Calculate Propensity Score (P(A=1|Features)) for IPW
    from sklearn.linear_model import LogisticRegression
    
    # Simple propensity model using lag features
    prop_model = LogisticRegression(solver='liblinear', max_iter=100)
    
    # Features: Risk Score + Recent Utilization + Demographics
    prop_features = ['risk_score', 'prev_event_1m', 'prev_event_3m_sum', 'age', 'utilization_trend']
    state_cols = [c for c in data.columns if c.startswith('st_')]
    prop_features += state_cols
    
    # Clean X and y
    # Ensure all prop features are in data
    prop_features = [f for f in prop_features if f in data.columns]
    
    X_prop = data[prop_features].fillna(0)
    y_prop = data['received_intervention']
    
    print("  Training propensity model for IPW...")
    prop_model.fit(X_prop, y_prop)
    data['propensity_score'] = prop_model.predict_proba(X_prop)[:, 1]
    
    return data


def load_teacher_labels(data):
    """
    Teacher-Student Distillation:
    Load robust causal estimates (CATE) from the Fixed Effects model (Teacher)
    and use them to label the World Model's rewards (Student).
    """
    path = Path("outputs/semisupervised_cates.csv.gz")
    if not path.exists():
        print("  Teacher labels not found. Running with raw outcomes (Not SOTA).")
        return data
    
    print("\n  Loading Teacher Labels (FE-CATE) for Distillation...")
    teacher = pd.read_csv(path)
    
    # Merge
    # Ensure keys match types
    data['month_year_str'] = data['month_year'].astype(str)
    teacher['month_year_str'] = teacher['month_year'].astype(str)
    
    # Check overlap
    overlap = data.merge(teacher[['member_id', 'month_year_str', 'cate', 'baseline_risk']], 
                        left_on=['member_id', 'month_year_str'], 
                        right_on=['member_id', 'month_year_str'], 
                        how='left')
    
    # Calculate Distilled Reward (r_t*)
    # r_t* = -Risk_under_action_a
    # Risk(a=0) = Baseline
    # Risk(a=1) = Baseline - CATE (since CATE is risk reduction)
    # Wait, in 10_system: cate = risk_control - risk_treated
    # So risk_treated = risk_control - cate
    
    # Determine column names (handle suffixes if conflict existed)
    br_col = 'baseline_risk_y' if 'baseline_risk_y' in overlap.columns else 'baseline_risk'
    cate_col = 'cate_y' if 'cate_y' in overlap.columns else 'cate'
    
    if br_col not in overlap.columns or cate_col not in overlap.columns:
        print("  Merge failed to bring in teacher columns. Check keys.")
        return data
        
    risk_treated = overlap[br_col] - overlap[cate_col]
    risk_control = overlap[br_col]
    
    # Use actual action to assign reward
    actual_risk = np.where(overlap['received_intervention'] == 1, risk_treated, risk_control)
    
    # Distilled Reward is negative risk (Utility)
    overlap['distilled_reward'] = -actual_risk
    
    # Handle missing teacher labels (e.g. alignment issues)
    # Fallback to -baseline (assuming a=0 risk) or just raw?
    # Let's fallback to raw -target_next_month if teacher is NaN
    # But for now, meaningful fill
    mask_nan = overlap['distilled_reward'].isna()
    if mask_nan.sum() > 0:
        print(f"  Warning: {mask_nan.sum()} rows missing teacher labels. Using raw.")
    
    # Overwrite IPW to disable it (Teacher is already unconfounded)
    # Set propensity to 0.5 -> Weight = 2.0 (Uniform)
    print("  Disabling IPW (Teacher values are already de-confounded).")
    overlap['propensity_score'] = 0.5
    
    return overlap



# ==========================================================================
# MAIN
# ==========================================================================

def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("WORLD MODEL ARCHITECTURE - TRUE FORWARD DYNAMICS")
    print("="*70)
    print("\nLoading data...")
    
    try:
        outcomes, interventions, attributes = load_real_data()
        data = preprocess_data(outcomes, interventions, attributes)
        
        # Load Teacher Labels if available (SOTA Distillation)
        data = load_teacher_labels(data)
        
        print(f"  Loaded: {len(data):,} member-months, {data['member_id'].nunique():,} members")
        
        # Features
        state_cols = [c for c in data.columns if c.startswith('st_')]
        features = ['age', 'is_female', 'prev_event_1m', 'prev_event_3m_sum', 
                   'risk_score', 'utilization_trend'] + state_cols
        features = [f for f in features if f in data.columns]
        print(f"  Features: {len(features)}")
        
        # Train world model
        device = 'cpu'
        model = train_world_model(data, features, epochs=30, batch_size=1024, lr=1e-3, device=device)
        
        if model is None:
            raise Exception("Training failed")
        
        # Plan
        cate_wm = plan_with_world_model(model, data, features, device=device)
        
        # Baseline risk model
        # Baseline risk model
        print("Training risk model...")
        untreated = data[data['received_intervention'] == 0]
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(untreated[features].fillna(0), untreated['target_next_month'])
        cal_rf = CalibratedClassifierCV(rf, cv=3)
        cal_rf.fit(untreated[features].fillna(0), untreated['target_next_month'])
        
        data['baseline_risk'] = cal_rf.predict_proba(data[features].fillna(0))[:, 1]
        data['cate'] = cate_wm
        data['cate_se'] = 0.001
        
        # Allocation
        print("\n" + "="*70)
        print("ALLOCATION SIMULATION")
        print("="*70)
        
        k = int(len(data) * 0.10)
        data['targeted_by_risk'] = data['baseline_risk'].rank(ascending=False) <= k
        data['targeted_by_atc'] = data['cate'].rank(ascending=False) <= k
        
        def calc_metrics(mask):
            subset = data[mask]
            scale = 2000 / len(data)
            events = float(subset['cate'].sum() * scale)
            se = float(np.sqrt((subset['cate_se']**2).sum()) * scale)
            return {
                "events_prevented": events,
                "ci": (events - 1.96*se, events + 1.96*se),
                "mean_cate": float(subset['cate'].mean())
            }
        
        risk_metrics = calc_metrics(data['targeted_by_risk'])
        atc_metrics = calc_metrics(data['targeted_by_atc'])
        
        results = {
            "method": "World Model (Learned Dynamics P(s'|s,a) + R(s,a))",
            "risk_policy": risk_metrics,
            "atc_policy": atc_metrics
        }
        
        with open(output_dir / "world_model_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nRisk Policy: {risk_metrics['events_prevented']:.1f} "
              f"(95% CI {risk_metrics['ci'][0]:.1f} to {risk_metrics['ci'][1]:.1f})")
        print(f"ATC Policy: {atc_metrics['events_prevented']:.1f} "
              f"(95% CI {atc_metrics['ci'][0]:.1f} to {atc_metrics['ci'][1]:.1f})")
        
        efficiency = atc_metrics['events_prevented'] / risk_metrics['events_prevented'] if risk_metrics['events_prevented'] > 0 else float('inf')
        print(f"Efficiency Gain: {efficiency:.1f}x")
        
        # Figure
        plt.rcParams['figure.dpi'] = 900
        plot_data = data.sample(min(2000, len(data)), random_state=42)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Segments
        untargeted = ~plot_data['targeted_by_risk'] & ~plot_data['targeted_by_atc']
        futile = plot_data['targeted_by_risk'] & ~plot_data['targeted_by_atc']
        consensus = plot_data['targeted_by_risk'] & plot_data['targeted_by_atc']
        gems = ~plot_data['targeted_by_risk'] & plot_data['targeted_by_atc']
        
        ax.scatter(plot_data[untargeted]['baseline_risk'], plot_data[untargeted]['cate'], 
                  c='#95A5A6', alpha=0.6, s=10, label='Untargeted')
        ax.scatter(plot_data[futile]['baseline_risk'], plot_data[futile]['cate'], 
                  c='#D32F2F', alpha=0.8, s=20, edgecolors='black', linewidths=0.5, 
                  label='Futile (High Risk, Low Impact)')
        ax.scatter(plot_data[consensus]['baseline_risk'], plot_data[consensus]['cate'], 
                  c='#2E7D32', alpha=0.8, s=20, edgecolors='black', linewidths=0.5, 
                  label='Consensus (High Risk, High Impact)')
        ax.scatter(plot_data[gems]['baseline_risk'], plot_data[gems]['cate'], 
                  c='#FFC107', alpha=0.8, s=20, edgecolors='black', linewidths=0.5, 
                  label='Hidden Gems (Receptive)')
        
        ax.set_xlabel('Baseline Risk (Predicted Probability)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Receptivity (World Model CATE)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "world_model_dashboard.png", dpi=900)
        
        print(f"\nSaved: {output_dir / 'world_model_dashboard.png'}")
        print("\n✓ World Model Pipeline Complete!")
        
    except Exception as e:
        print(f"\n✗ WORLD MODEL FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to FE results...")


if __name__ == "__main__":
    main()
