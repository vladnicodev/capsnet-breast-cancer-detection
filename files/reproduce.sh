#!/usr/bin/env bash
# Reproduce every number and figure in the README (~70 min on an RTX 5080).
#   bash reproduce.sh /path/to/pcam     # folder containing train+val/{train,valid}/{0,1}
set -euo pipefail
PCAM_ROOT=${1:?usage: bash reproduce.sh /path/to/pcam}

python -m capsnet_pcam.prepare_data --src "$PCAM_ROOT" --out data/pcam

# Main grid: 3 models x 5 training-set sizes x 3 seeds = 45 runs
python -m capsnet_pcam.sweep

# CapsNet ablations: no routing (1 iteration = uniform coupling) and no reconstruction decoder
python -m capsnet_pcam.sweep --models capsnet --fractions 0.01 0.1 1.0 --variant no-routing --routings 1
python -m capsnet_pcam.sweep --models capsnet --fractions 0.01 0.1 1.0 --variant no-recon --recon-weight 0

# Test-time distribution shifts on the models trained on all data
python -m capsnet_pcam.robustness --fractions 1.0

# Tables (results/summary.md) and figures (results/figures/)
python -m capsnet_pcam.report
