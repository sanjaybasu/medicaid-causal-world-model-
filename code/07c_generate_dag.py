"""
Step 07c: Generate Causal DAG
=============================
Generates eFigure 1: Directed Acyclic Graph (DAG) for Identification Strategy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_dag():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=900)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Nodes
    # U: Unmeasured Confounders
    # H: History
    # S: State (derived from H)
    # A: Action
    # Y: Outcome
    
    nodes = {
        'U': (5, 7),
        'H': (2, 4),
        'S': (4, 4),
        'A': (6, 4),
        'Y': (8, 4)
    }
    
    # Draw Nodes
    for name, (x, y) in nodes.items():
        circle = patches.Circle((x, y), radius=0.6, fc='white', ec='black', lw=2)
        ax.add_patch(circle)
        label = name if name != 'U' else 'U'
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold')
        
    # Edges
    # Use shrinkA/shrinkB to avoid overlap with nodes (Radius 0.6)
    # 0.6 data units in 8x6 fig (approx 72 dpi * 0.6 = ~45 points) usually.
    # But shrink is in points. Let's try shrink=15.
    
    arrow_opts = dict(arrowstyle="->", lw=2, shrinkA=15, shrinkB=15)
    
    ax.annotate("", xy=nodes['S'], xytext=nodes['H'], arrowprops=arrow_opts)
    ax.annotate("", xy=nodes['A'], xytext=nodes['S'], arrowprops=arrow_opts)
    # S->Y using arc and shrink
    ax.annotate("", xy=nodes['Y'], xytext=nodes['S'], 
                arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.5", shrinkA=15, shrinkB=15))
    
    ax.annotate("", xy=nodes['Y'], xytext=nodes['A'], arrowprops=arrow_opts)
    
    # U -> Y (Dashed)
    ax.annotate("", xy=nodes['Y'], xytext=nodes['U'], 
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="--", color="gray", shrinkA=15, shrinkB=15))
    # U -> A (Dashed)
    ax.annotate("", xy=nodes['A'], xytext=nodes['U'], 
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="--", color="gray", shrinkA=15, shrinkB=15))
    
    # Legend
    ax.text(5, 1, "Assumption: $Y(a) \perp A | S$\n(Sufficient Adjustment)", ha='center', fontsize=12)
    
    output_path = "outputs/efigure1_dag.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved DAG to {output_path}")

if __name__ == "__main__":
    draw_dag()
