"""
Step 07b: Generate Nature-Quality Architecture Diagram
======================================================
Generates Figure 1: Deep Causal World Model Architecture.
Optimized layout with clean routing and professional styling.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_neural_architecture():
    # Settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(14, 10), dpi=900)
    # Ax limits extended to allow left-side routing
    ax.set_xlim(-1.5, 14.5)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # --- Style Constants ---
    # "Nature" palette: Soft but distinct
    blue_light = "#E3F2FD"
    blue_stroke = "#1E88E5"
    blue_text = "#0D47A1"
    
    orange_light = "#FFF3E0"
    orange_stroke = "#FB8C00"
    orange_text = "#E65100"
    
    purple_fill = "#F3E5F5"
    purple_stroke = "#8E24AA"
    
    green_fill = "#E8F5E9"
    green_stroke = "#43A047"
    green_text = "#1B5E20"
    
    gray_label = "#757575"
    
    # Common box props
    def box(x, y, w, h, fc, ec, label=None, sublabel=None, zorder=10):
        # Shadow
        shadow = FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.2", fc="#E0E0E0", ec="none", zorder=zorder-2)
        ax.add_patch(shadow)
        # Main Box
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", fc=fc, ec=ec, lw=2, zorder=zorder-1)
        ax.add_patch(p)
        
        # Text
        cx = x + w/2
        cy = y + h/2
        if label:
            ax.text(cx, cy + (0.15 if sublabel else 0), label, ha='center', va='center', fontweight='bold', fontsize=11, color=ec, zorder=zorder)
        if sublabel:
            ax.text(cx, cy - 0.15, sublabel, ha='center', va='center', fontsize=9, color=ec, zorder=zorder)
            
    # ==========================
    # LAYER LABELS (Left Column)
    # ==========================
    ax.text(-1, 9.0, "INPUTS", fontsize=10, fontweight='bold', color=gray_label, va='center', ha='right')
    ax.text(-1, 6.5, "ENCODING", fontsize=10, fontweight='bold', color=gray_label, va='center', ha='right')
    ax.text(-1, 3.5, "LATENT\nDYNAMICS", fontsize=10, fontweight='bold', color=gray_label, va='center', ha='right')
    ax.text(-1, 1.2, "POLICY", fontsize=10, fontweight='bold', color=gray_label, va='center', ha='right')
    
    # Grid lines (Subtle)
    for y in [7.8, 5.2, 2.2]:
        ax.plot([-1, 14], [y, y], color="#EEEEEE", lw=1, zorder=0)

    # ==========================
    # LAYER 1: DATA INPUTS
    # ==========================
    # Structured Data
    box(1.5, 8.2, 3.5, 1.2, blue_light, blue_stroke, "Patient History", "Structured: Utilization, Demographics")
    
    # Unstructured Data
    box(9.0, 8.2, 3.5, 1.2, orange_light, orange_stroke, "Clinical Notes", "Unstructured: Text Logs")
    
    # ==========================
    # LAYER 2: ENCODERS
    # ==========================
    # Arrows Down
    ax.annotate("", xy=(3.25, 7.5), xytext=(3.25, 8.2), arrowprops=dict(arrowstyle="->", lw=2, color=blue_stroke))
    ax.text(3.35, 7.8, "Normalize", fontsize=8, color=blue_stroke)
    
    ax.annotate("", xy=(10.75, 7.5), xytext=(10.75, 8.2), arrowprops=dict(arrowstyle="->", lw=2, color=orange_stroke))
    ax.text(10.85, 7.8, "Tokenize", fontsize=8, color=orange_stroke)
    
    # Encoder Boxes
    box(2.0, 6.0, 2.5, 1.2, "white", blue_stroke, "Feature Encoder", "MLP Layers")
    box(9.0, 5.8, 3.5, 1.4, "white", orange_stroke, "Language Model", "Gemini Pro 3 Teacher / DistilBERT Student")
    
    # ==========================
    # LAYER 3: WORLD MODEL
    # ==========================
    # State Convergence Arrows
    # Curve from Encoder to State
    ax.annotate("", xy=(5.5, 4.8), xytext=(3.25, 6.0), 
                arrowprops=dict(arrowstyle="->", lw=2, color=blue_stroke, connectionstyle="arc3,rad=0.1"))
    
    ax.annotate("", xy=(8.5, 4.8), xytext=(10.75, 5.8), 
                arrowprops=dict(arrowstyle="->", lw=2, color=orange_stroke, connectionstyle="arc3,rad=-0.1"))
    
    # Labels on arrows
    ax.text(3.8, 5.2, "State $S_t$", fontsize=10, fontweight='bold', color=blue_text, bbox=dict(fc='white', ec='none', pad=1))
    ax.text(9.5, 5.2, "Action $A_t$", fontsize=10, fontweight='bold', color=orange_text, bbox=dict(fc='white', ec='none', pad=1))
    
    # Main Box: Causal World Model
    # Dimensions
    wm_x, wm_y, wm_w, wm_h = 4.5, 2.5, 5.0, 2.3
    # Shadow
    ax.add_patch(FancyBboxPatch((wm_x+0.05, wm_y-0.05), wm_w, wm_h, boxstyle="round,pad=0.2", fc="#CABEFF", ec="none", zorder=1))
    # Main
    ax.add_patch(FancyBboxPatch((wm_x, wm_y), wm_w, wm_h, boxstyle="round,pad=0.2", fc=purple_fill, ec=purple_stroke, lw=2.5, zorder=2))
    ax.text(wm_x + wm_w/2, wm_y + 2.0, "CAUSAL WORLD MODEL", ha='center', fontweight='bold', color=purple_stroke, zorder=3)
    
    # Internal Modules
    box(4.8, 2.8, 2.0, 1.0, "white", purple_stroke, "Transition", "$P(S'|S,A)$", zorder=4)
    box(7.2, 2.8, 2.0, 1.0, "white", purple_stroke, "Reward", "$R(S,A)$", zorder=4)
    
    # ==========================
    # LAYER 4: POLICY
    # ==========================
    ax.annotate("", xy=(7.0, 2.2), xytext=(7.0, 2.5), arrowprops=dict(arrowstyle="->", lw=3, color=purple_stroke))
    
    box(5.5, 0.5, 3.0, 1.5, green_fill, green_stroke, "OPTIMAL POLICY $\pi^*$", "Batch-Constrained Q-Learning\n(Air Traffic Control)", zorder=10)
    
    # ==========================
    # FEEDBACK LOOP (Dashed Arrow)
    # ==========================
    # Route: From Transition (4.8, 3.3) -> Left -> Up -> In to History
    # Start point: Left edge of Transition box (approx 4.8, 3.3)
    # Actually, transition box is at x=4.8.
    
    start_point = (4.8, 3.3)
    # Route points
    p1 = (0.0, 3.3)   # Far Left
    p2 = (0.0, 8.8)   # Top Level
    p3 = (1.5, 8.8)   # Into Patient History
    
    path_verts = [start_point, p1, p2, p3]
    path_codes = [patches.Path.MOVETO, patches.Path.LINETO, patches.Path.LINETO, patches.Path.LINETO]
    
    path = patches.Path(path_verts, path_codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor="#7F8C8D", lw=2, linestyle='--', zorder=0)
    ax.add_patch(patch)
    
    # Arrow head at end
    ax.annotate("", xy=p3, xytext=(0.0, 8.8), arrowprops=dict(arrowstyle="->", lw=2, color="#7F8C8D"))
    
    # Label for feedback
    ax.text(0.2, 6.0, "Next State Feedback $(S_{t+1})$", rotation=90, color="#7F8C8D", fontsize=9, va='center', bbox=dict(fc='white', ec='none'))
    
    # ==========================
    # LEGEND
    # ==========================
    leg_x, leg_y = 12.0, 0.5
    ax.add_patch(patches.Rectangle((leg_x-0.2, leg_y-0.2), 2.2, 2.5, fc="white", ec="#CCC", lw=1, zorder=5))
    ax.text(leg_x+0.9, leg_y+2.0, "Legend", ha='center', fontweight='bold', color="#555")
    
    items = [
        (blue_light, blue_stroke, "Structured Data"),
        (orange_light, orange_stroke, "Unstructured Data"),
        (purple_fill, purple_stroke, "Causal Model"),
        (green_fill, green_stroke, "Optimal Policy")
    ]
    
    for i, (fc, ec, lbl) in enumerate(items):
        y_pos = leg_y + 1.5 - (i * 0.4)
        ax.add_patch(patches.Rectangle((leg_x, y_pos), 0.3, 0.25, fc=fc, ec=ec))
        ax.text(leg_x + 0.4, y_pos + 0.1, lbl, fontsize=9, va='center')

    plt.tight_layout()
    output_path = "outputs/figure1_study_flow.png"
    plt.savefig(output_path, dpi=900, bbox_inches='tight')
    print(f"Saved optimized diagram to {output_path}")

if __name__ == "__main__":
    draw_neural_architecture()
