"""
Step 07b: Generate Nature-Quality Architecture Diagram
======================================================
Generates Figure 1: Deep Causal World Model Architecture.
Style adapted from 'healthcare_world_model' reference (Dual Stream Neural Network).

Layout:
- Top: Inputs (Structured Data + Unstructured Clinical Notes)
- Middle: Encoders (Feature MLP + DistilBERT -> Action Space)
- Core: State Representation & World Model Dynamics
- Bottom: RL Policy & Outcome Heads

Colors:
- Patient Data: Blue (#E3F2FD / #1976D2)
- Clinical Text/Actions: Orange (#FFF3E0 / #E65100)
- Core Model: Gray/Purple
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_neural_architecture():
    # Settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=900)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # --- Style Constants ---
    blue_light = "#E3F2FD"
    blue_dark = "#1565C0"
    orange_light = "#FFF3E0"
    orange_dark = "#E65100"
    gray_light = "#F5F5F5"
    gray_dark = "#424242"
    
    bbox_props = dict(boxstyle="round,pad=0.3", lw=2)
    
    # ==========================
    # LAYER 1: INPUTS (Top)
    # ==========================
    ax.text(1, 9.5, "INPUT LAYER", fontsize=10, fontweight='bold', color="#777")
    
    # Box 1: Patient History (Structured)
    ax.add_patch(FancyBboxPatch((1, 8), 3.5, 1.2, fc=blue_light, ec=blue_dark, **bbox_props))
    ax.text(2.75, 8.6, "Patient History\n(Structured)", ha='center', va='center', fontweight='bold', color=blue_dark)
    ax.text(2.75, 8.3, "Demographics, Utilization", ha='center', va='center', fontsize=9, color=blue_dark)
    
    # Box 2: Clinical Notes (Unstructured)
    ax.add_patch(FancyBboxPatch((9.5, 8), 3.5, 1.2, fc=orange_light, ec=orange_dark, **bbox_props))
    ax.text(11.25, 8.6, "Clinical Notes\n(Unstructured Text)", ha='center', va='center', fontweight='bold', color=orange_dark)
    ax.text(11.25, 8.3, "Care Team Logs, SOAP Notes", ha='center', va='center', fontsize=9, color=orange_dark)
    
    # ==========================
    # LAYER 2: ENCODERS
    # ==========================
    ax.text(1, 7.5, "ENCODER LAYER", fontsize=10, fontweight='bold', color="#777")
    
    # Arrow 1 down
    ax.annotate("", xy=(2.75, 7.2), xytext=(2.75, 8.0), arrowprops=dict(arrowstyle="->", lw=2, color=blue_dark))
    ax.text(2.85, 7.6, "Norm", fontsize=8, color=blue_dark)
    
    # Arrow 2 down
    ax.annotate("", xy=(11.25, 7.2), xytext=(11.25, 8.0), arrowprops=dict(arrowstyle="->", lw=2, color=orange_dark))
    ax.text(11.35, 7.6, "Tokenize", fontsize=8, color=orange_dark)
    
    # Encoder 1: Feature MLP
    ax.add_patch(FancyBboxPatch((1.5, 6.0), 2.5, 1.2, fc="white", ec=blue_dark, boxstyle="round,pad=0.1", lw=2))
    ax.text(2.75, 6.6, "Feature Encoder\n(MLP)", ha='center', va='center', fontweight='bold', color=blue_dark)
    
    # Encoder 2: NLP Model
    ax.add_patch(FancyBboxPatch((9.5, 5.5), 3.5, 1.7, fc="white", ec=orange_dark, boxstyle="round,pad=0.1", lw=2))
    ax.text(11.25, 6.8, "Intervention Extraction", ha='center', va='center', fontweight='bold', color=orange_dark)
    # Inner block: LLM
    ax.add_patch(FancyBboxPatch((10.0, 5.7), 2.5, 0.8, fc=orange_light, ec=orange_dark, boxstyle="round,pad=0.1"))
    ax.text(11.25, 6.1, "GPT-4 Teacher\nDistilBERT Student", ha='center', va='center', fontsize=9, color=orange_dark)
    
    # ==========================
    # LAYER 3: LATENT STATE
    # ==========================
    ax.text(1, 5.0, "LATENT REPRESENTATION", fontsize=10, fontweight='bold', color="#777")
    
    # Arrows converging
    ax.annotate("", xy=(5.5, 4.8), xytext=(2.75, 6.0), arrowprops=dict(arrowstyle="->", lw=2, color=blue_dark, connectionstyle="arc3,rad=0.1"))
    ax.text(3.5, 5.2, "State $S_t$", fontsize=10, color=blue_dark, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none'))
    
    ax.annotate("", xy=(8.5, 4.8), xytext=(11.25, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color=orange_dark, connectionstyle="arc3,rad=-0.1"))
    ax.text(10.0, 5.2, "Action $A_t$\n(44-dim)", fontsize=10, color=orange_dark, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none'))
    
    # Central Core: World Model (MDP)
    # Large rounded box
    ax.add_patch(FancyBboxPatch((4.5, 2.5), 5.0, 2.3, boxstyle="round,pad=0.2", fc="#F3E5F5", ec="#8E24AA", lw=3))
    ax.text(7.0, 4.5, "CAUSAL WORLD MODEL", ha='center', va='center', fontweight='bold', fontsize=12, color="#8E24AA")
    
    # Internal: Transition
    ax.add_patch(FancyBboxPatch((4.8, 2.8), 2.0, 1.2, fc="white", ec="#8E24AA", boxstyle="round,pad=0.1"))
    ax.text(5.8, 3.4, "Transition\nDynamics", ha='center', va='center', fontweight='bold', fontsize=10, color="#8E24AA")
    ax.text(5.8, 3.0, "$P(S'|S,A)$", ha='center', va='center', fontsize=9)
    
    # Internal: Reward
    ax.add_patch(FancyBboxPatch((7.2, 2.8), 2.0, 1.2, fc="white", ec="#8E24AA", boxstyle="round,pad=0.1"))
    ax.text(8.2, 3.4, "Reward\nFunction", ha='center', va='center', fontweight='bold', fontsize=10, color="#8E24AA")
    ax.text(8.2, 3.0, "$R(S,A)$", ha='center', va='center', fontsize=9)
    
    # ==========================
    # LAYER 4: OUPUTS (Bottom)
    # ==========================
    ax.text(1, 1.5, "OUTPUT & POLICY", fontsize=10, fontweight='bold', color="#777")
    
    # Arrow down
    ax.annotate("", xy=(7.0, 2.0), xytext=(7.0, 2.5), arrowprops=dict(arrowstyle="->", lw=3, color="#8E24AA"))
    
    # Policy Head
    ax.add_patch(FancyBboxPatch((5.5, 0.5), 3.0, 1.5, fc="#E8F5E9", ec="#2E7D32", boxstyle="round,pad=0.2", lw=2))
    ax.text(7.0, 1.6, "OPTIMAL POLICY $\pi^*$", ha='center', va='center', fontweight='bold', fontsize=11, color="#2E7D32")
    ax.text(7.0, 1.2, "Batch-Constrained\nQ-Learning (BCQ)", ha='center', va='center', fontsize=10, color="#2E7D32")
    ax.text(7.0, 0.8, "\"Air Traffic Control\"", ha='center', va='center', fontsize=9, fontstyle='italic', color="#2E7D32")
    
    # Feedback loop visualization (Curved arrow from Bottom to Top Left)
    # Using a large arc
    path = patches.Path(
        [(5.5, 1.25), (1.0, 1.25), (1.0, 6.6), (1.5, 6.6)],
        [patches.Path.MOVETO, patches.Path.CURVE4, patches.Path.CURVE4, patches.Path.CURVE4]
    )
    # patch = patches.PathPatch(path, facecolor='none', edgecolor='#999', lw=1.5, linestyle='--')
    # ax.add_patch(patch)
    # Arrow is hard with Path. Using fancy annotation.
    
    ax.annotate("", xy=(1.0, 8.6), xytext=(5.5, 1.25), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#7F8C8D", connectionstyle="bar,fraction=-0.2,angle=180", linestyle="--"))
    ax.text(0.5, 5.0, "Next State\n(Feedback)", ha='right', va='center', fontsize=9, color="#7F8C8D", rotation=90)
    
    # Legend
    rect = patches.Rectangle((11, 0.5), 2.5, 2.0, linewidth=1, edgecolor='#ccc', facecolor='#fff')
    ax.add_patch(rect)
    ax.text(12.25, 2.3, "Legend", ha='center', fontweight='bold')
    
    # Legend Items
    ax.add_patch(patches.Rectangle((11.2, 1.9), 0.2, 0.2, fc=blue_light, ec=blue_dark))
    ax.text(11.5, 2.0, "Patient State", fontsize=9, va='center')
    
    ax.add_patch(patches.Rectangle((11.2, 1.5), 0.2, 0.2, fc=orange_light, ec=orange_dark))
    ax.text(11.5, 1.6, "Intervention/Action", fontsize=9, va='center')
    
    ax.add_patch(patches.Rectangle((11.2, 1.1), 0.2, 0.2, fc="#F3E5F5", ec="#8E24AA"))
    ax.text(11.5, 1.2, "World Model", fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig("outputs/figure1_study_flow.png", bbox_inches='tight', dpi=900)
    print("Saved Deep Architecture Figure 1 to outputs/figure1_study_flow.png")
    
if __name__ == "__main__":
    draw_neural_architecture()
