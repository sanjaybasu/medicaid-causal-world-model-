#!/usr/bin/env bash
# Build .docx artifacts for PHM submission.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Build a manuscript-with-embedded-figures version by appending image
#    references to the manuscript, then run pandoc.
TMP=$(mktemp -t phm_manuscript_with_figs.XXXX.md)
trap 'rm -f "$TMP"' EXIT

# Construct manuscript with figures inline (after legends).
python3 <<'PY' > "$TMP"
from pathlib import Path
text = Path("manuscript.md").read_text()

# Insert figures right before "## Acknowledgments"
fig_block = """\

---

## Figures (embedded)

![Figure 1. Analytic schema.](figures/Figure_1_Schema.png)

*Figure 1. Analytic schema. Each calendar month t, every beneficiary is described by a 127-feature covariate vector X_{it} including demographics, prior utilization, chronic-condition indicators, care-management engagement history, and temporal markers. Cross-fitted nuisance functions estimate the propensity ê(x) and outcome regressions μ̂_a(x). A causal forest combines these into the conditional average treatment effect τ̂(x). Two monthly allocation rules are compared at fixed 10% capacity: π_R contacts the top-decile by predicted risk; π_E contacts the top-decile by predicted CATE. Policy values are estimated via doubly robust off-policy evaluation. Identification add-ons (within-person fixed effects, MSM with stabilized IPTW, target-trial emulation, staggered-rollout IV) provide cross-validation across causal-identification strategies.*

![Figure 2. CATE heterogeneity and primary contrast.](figures/Figure_2_CATE_and_Contrast.png)

*Figure 2. Distribution of estimated CATEs and primary contrast. (A) Histogram of 2,670,806 cross-fitted causal-forest CATE estimates; mean −0.025; 87.8% of person-months had τ̂ < 0. (B) Acute events prevented per 2,000 members per month under risk-based (π_R, blue) and effect-based (π_E, orange) allocation, both at fixed 10% capacity. Bars are point estimates from doubly robust off-policy evaluation; whiskers are 95% confidence intervals from 1,000-replicate individual-level cluster bootstrap.*

![Figure 3. Capacity-free contrast and quasi-experimental validation.](figures/Figure_3_RATE_and_IV.png)

*Figure 3. Capacity-free contrast and quasi-experimental validation. (A) Rank-weighted average treatment effect (RATE) curve from 1% to 50% capacity. The effect-based rule (orange) lies strictly above the risk-based rule (blue) at every capacity. AUTC = 0.31 (95% CI, 0.28 to 0.34). (B) 30-day acute-event rate by predicted-CATE tertile in the IV-eligible subsample (n = 24,997 beneficiaries straddling staggered rollout). Higher tertiles experienced lower outcome rates (low: 46.2%; medium: 27.2%; high: 25.1%; P < .001), consistent with the observational CATE estimates.*

"""

# Splice before "## Acknowledgments"
marker = "## Acknowledgments"
idx = text.index(marker)
spliced = text[:idx] + fig_block + text[idx:]
print(spliced)
PY

# 2. Convert manuscript with embedded figures.
pandoc "$TMP" \
  -o manuscript.docx \
  --standalone \
  --reference-doc=$(pwd)/.pandoc_reference.docx 2>/dev/null \
  || pandoc "$TMP" -o manuscript.docx --standalone

# 3. Convert appendix.
pandoc appendix.md -o appendix.docx --standalone

# 4. Convert cover letter.
pandoc cover_letter.md -o cover_letter.docx --standalone

ls -lh manuscript.docx appendix.docx cover_letter.docx
echo "Built."
