"""
Step 8: Compile Submission Package
==================================
Compiles the manuscript, figures, and appendix into a single PDF.
Uses Pandoc and XeLaTeX.

This script:
1. Reads markdown files.
2. Resolves image paths to absolute paths.
3. Combines content.
4. Generates PDF.
"""

import subprocess
from pathlib import Path
import os
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compile_pdf():
    print("Compiling submission package...")
    
    config = load_config()
    base_dir = Path(os.getcwd())
    
    manuscript_path = base_dir / "manuscript_lancet.md"
    appendix_path = base_dir / "appendix_lancet.md"
    output_pdf = base_dir / "submission_package.pdf"
    
    # Define Image Paths
    # Main Text Figures
    fig1 = base_dir / "outputs/figure1_study_flow.png"
    fig2 = base_dir / "outputs/figure3_receptivity_window.png" # Renumbered Fig 3 -> Fig 2
    fig3 = base_dir / "outputs/figure4_patient_trajectory.png" # Renumbered Fig 4 -> Fig 3
    
    # Appendix Figures
    efig2 = base_dir / "outputs/figure2_policy_performance.png" # Moved Fig 2 -> eFig 2
    
    # Verify images exist
    for p in [fig1, fig2, fig3, efig2]:
        if not p.exists():
            print(f"Error: Image not found at {p}")
            # return # Do not fail, just warn for now to debug
            
    # Read content
    with open(manuscript_path, 'r') as f:
        manuscript_text = f.read()
        
    with open(appendix_path, 'r') as f:
        appendix_text = f.read()
        
    # Replace unicode characters that might choke latex
    manuscript_text = manuscript_text.replace("≥", ">=")
    appendix_text = appendix_text.replace("≥", ">=")
    
    # Resolving Appendix Image Paths
    appendix_text = appendix_text.replace("EFIG2_PATH", str(efig2))
    
    # Constuct combined markdown
    # Insert figures after manuscript, before appendix
    combined_text = f"""{manuscript_text}

\\newpage

# Figures

![**Figure 1. Causal World Model Architecture.** The care delivery system is modeled as a Markov Decision Process. (A) States ($S_t$) capture patient trajectory (demographics + utilization history). (B) Actions ($A_t$) are interventions extracted from clinical notes. (C) Rewards ($R_t$) are avoidance of acute events. (D) The model learns a policy $\\pi(S_t)$ to maximize long-term rewards.]({fig1})

![**Figure 2. Heterogeneity of Treatment Effects.** Conditional Average Treatment Effects (CATE) of phone outreach stratified by receptivity state. The x-axis represents receptivity tiers defined by recent ED utilization. The y-axis shows the estimated reduction in acute event probability. Efficacy is non-linear: treatments are highly effective in the high-receptivity window (NNT 1.1) but futile in the low-receptivity state (NNT > 3000), illustrating state-dependent causal mechanisms.]({fig2})

![**Figure 3. Within-Person Receptivity Windows: A Longitudinal Case Study.** A representative patient trajectory (365 days) illustrating the dynamic nature of receptivity. The blue line represents the estimated CATE (efficacy) of intervention. Red markers indicate acute events (ED visits), which trigger "receptivity windows" (shaded blue) where efficacy spikes. A standard intervention delivered at Day 100 (grey dot) occurs during a stable period and has low impact. An optimal intervention delivered at Day 182 (green star), inside a window, achieves maximal impact. This variability demonstrates that "high risk" is a state, not a trait.]({fig3})

\\newpage

{appendix_text}
"""

    # Write temp file
    temp_md = base_dir / "submission_temp_absolute.md"
    with open(temp_md, 'w') as f:
        f.write(combined_text)
    print(f"Created temporary markdown: {temp_md}")
        
    # Run Pandoc
    # Requires texlive-xetex for XeLaTeX
    cmd = [
        "pandoc",
        str(temp_md),
        "-o", str(output_pdf),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=Helvetica"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Success! PDF generated at: {output_pdf}")
    except subprocess.CalledProcessError as e:
        print(f"Error during PDF compilation: {e}")

if __name__ == "__main__":
    compile_pdf()
