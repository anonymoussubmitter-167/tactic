#!/bin/bash
# Retrain v1 and v2 models with full logging
# v3 already exists in checkpoints/v3/

set -e

cd /home/bcheng/tactic

echo "=============================================="
echo "TACTIC-Kinetics Full Retraining"
echo "=============================================="

# ============================================
# v1: Basic multi-condition (5 conditions)
# ============================================
echo ""
echo "=============================================="
echo "Training v1: 5 conditions, basic strategies"
echo "=============================================="

python train.py \
    --epochs 100 \
    --batch-size 32 \
    --n-conditions 5 \
    --samples-per-mech 1000 \
    --dataset-path data/multi_condition_v1.pt \
    --checkpoint-dir checkpoints/v1 \
    --log-dir logs/v1 \
    --seed 42 \
    --regenerate

echo "v1 training complete!"

# ============================================
# v2: Improved multi-condition (20 conditions)
# ============================================
echo ""
echo "=============================================="
echo "Training v2: 20 conditions, improved strategies"
echo "=============================================="

python train.py \
    --epochs 100 \
    --batch-size 32 \
    --n-conditions 20 \
    --samples-per-mech 1000 \
    --dataset-path data/multi_condition_v2.pt \
    --checkpoint-dir checkpoints/v2 \
    --log-dir logs/v2 \
    --seed 42

echo "v2 training complete!"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "All training complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  v1: checkpoints/v1/, logs/v1/"
echo "  v2: checkpoints/v2/, logs/v2/"
echo "  v3: checkpoints/v3/, logs/v3/ (already exists)"
echo ""
echo "Run comparison with:"
echo "  python scripts/compare_all_versions.py"
