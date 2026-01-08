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
    edges = [
        ('H', 'S', 'solid'),
        ('S', 'A', 'solid'),
        ('S', 'Y', 'solid'),
        ('A', 'Y', 'solid'),
        ('U', 'Y', 'dashed'), # Unmeasured confounding on Y
        ('U', 'A', 'dashed')  # Unmeasured confounding on A? Assuming assumption of No Unmeasured Confounding holds for S.
                              # If S satisfies backdoor, no U->A arrow if conditioned on S?
                              # The paper assumes Selection on Observables (S).
                              # So U->A is blocked by S? No, U is unmeasured.
                              # If Unconfoundedness holds, there is no U that points to both A and Y.
                              # But in reality U exists. We assume S blocks H->A.
                              # Let's draw the "Assumed" DAG where S is sufficient.
    ]
    
    # Identification Assumption DAG
    # H -> S
    # S -> A (Policy)
    # S -> Y (Prognosis)
    # A -> Y (Effect)
    # U -> Y (Noise)
    # U -> A (Violation! We assume this is absent or weak, hence E-value analysis).
    # I will draw the standard identifying DAG.
    
    ax.annotate("", xy=nodes['S'], xytext=(2.6, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=nodes['A'], xytext=(4.6, 4), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=nodes['Y'], xytext=(6.6, 4), arrowprops=dict(arrowstyle="->", lw=2))
    
    # S -> Y (Confounding path control)
    # Arc
    ax.annotate("", xy=nodes['Y'], xytext=nodes['S'], arrowprops=dict(arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.5"))
    
    # U -> Y
    ax.annotate("", xy=(8, 4.6), xytext=(5.2, 6.8), arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="--", color="gray"))
    # U -> A (The threat)
    ax.annotate("", xy=(6, 4.6), xytext=(4.8, 6.8), arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="--", color="gray"))
    
    # Legend
    ax.text(5, 1, "Assumption: $Y(a) \perp A | S$\n(Sufficient Adjustment)", ha='center', fontsize=12)
    
    output_path = "outputs/efigure1_dag.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved DAG to {output_path}")

if __name__ == "__main__":
    draw_dag()
