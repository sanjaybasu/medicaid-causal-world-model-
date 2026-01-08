"""
Step 11: Package Submission
===========================
Organize the final submission folder and clean up workspace.

This script:
1. Creates 'submission/' directory structure.
2. Renames and moves final Manipulate/Appendix.
3. Renames and moves Figures to 'submission/figures/'.
4. Archives outdated files.

Outputs:
    - submission/
        - Main_Manuscript.md
        - Supplementary_Appendix.md
        - figures/
            - Figure_1.png
            - Figure_2.png
            - Figure_3.png
            - Figure_4.png
            - eFigure_2.png
"""

import shutil
import os
from pathlib import Path

def package_submission():
    print("Packaging submission...")
    base_dir = Path(os.getcwd())
    sub_dir = base_dir / "submission"
    fig_dir = sub_dir / "figures"
    archive_dir = base_dir / "_archive"
    
    # Create directories
    if sub_dir.exists():
        shutil.rmtree(sub_dir)
    sub_dir.mkdir()
    fig_dir.mkdir()
    
    if not archive_dir.exists():
        archive_dir.mkdir()
        
    # 1. Copy Manuscripts
    print("Copying manuscripts...")
    shutil.copy(base_dir / "manuscript_lancet.md", sub_dir / "Main_Manuscript.md")
    shutil.copy(base_dir / "appendix_lancet.md", sub_dir / "Supplementary_Appendix.md")
    
    # 2. Copy Figures (Renaming for logical consistency)
    print("Copying figures...")
    
    # Map: Source -> Destination
    # Manuscript Figure 1 = Study Flow
    # Manuscript Figure 2 = Heterogeneity (was figure3_receptivity)
    # Manuscript Figure 3 = Longitudinal (was figure4_patient)
    # Manuscript Figure 4 = ATC Dashboard (was figure5_atc)
    # Appendix eFigure 2 = Ablation (was figure2_policy)
    
    mapping = {
        "outputs/figure1_study_flow.png": "Figure_1.png",
        "outputs/figure3_receptivity_window.png": "Figure_2.png",
        "outputs/figure4_patient_trajectory.png": "Figure_3.png",
        "outputs/figure5_atc_dashboard.png": "Figure_4.png",
        "outputs/figure2_policy_performance.png": "eFigure_2.png"
    }
    
    for src, dst in mapping.items():
        src_path = base_dir / src
        dst_path = fig_dir / dst
        if src_path.exists():
            shutil.copy(src_path, dst_path)
            print(f"  Copied {src} -> {dst}")
        else:
            print(f"  WARNING: Source figure not found: {src}")
            
    # 3. Archive Outdated Files
    print("Archiving outdated files...")
    # List of patterns to archive
    patterns = ["manuscript_*.md", "appendix_*.md", "paper_draft*.md", "results.md"]
    keep_list = ["manuscript_lancet.md", "appendix_lancet.md"] # Keep current source of truth in root for now
    
    for pattern in patterns:
        for file in base_dir.glob(pattern):
            if file.name not in keep_list:
                print(f"  Archiving {file.name}")
                shutil.move(str(file), str(archive_dir / file.name))
                
    print("Submission package created successfully at 'submission/'")

if __name__ == "__main__":
    package_submission()
