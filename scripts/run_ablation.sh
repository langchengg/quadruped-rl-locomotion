#!/bin/bash
# ──────────────────────────────────────────────────
# Reward Ablation Experiment Runner
# 
# Runs baseline + 6 ablation experiments sequentially.
# Each experiment trains for a reduced number of steps
# (500K) for quick comparison.
#
# Usage: bash scripts/run_ablation.sh
# ──────────────────────────────────────────────────

set -e

TIMESTEPS=500000
NUM_ENVS=4
DEVICE="auto"

echo "══════════════════════════════════════════════"
echo "  Reward Ablation Study"
echo "  Timesteps per experiment: ${TIMESTEPS}"
echo "══════════════════════════════════════════════"

# Baseline: all rewards active
echo -e "\n[1/7] Training baseline (all rewards)..."
python src/train.py \
    --algo ppo \
    --total_timesteps $TIMESTEPS \
    --num_envs $NUM_ENVS \
    --exp_name ablation_baseline \
    --device $DEVICE \
    --seed 42

# Ablation experiments
declare -A ABLATIONS
ABLATIONS[tracking_lin_vel]="tracking_lin_vel:0.0"
ABLATIONS[tracking_ang_vel]="tracking_ang_vel:0.0"
ABLATIONS[lin_vel_z]="lin_vel_z:0.0"
ABLATIONS[base_height]="base_height:0.0"
ABLATIONS[action_rate]="action_rate:0.0"
ABLATIONS[similar_to_default]="similar_to_default:0.0"

COUNT=2
for NAME in "${!ABLATIONS[@]}"; do
    echo -e "\n[${COUNT}/7] Training ablation: no_${NAME}..."
    python src/train.py \
        --algo ppo \
        --total_timesteps $TIMESTEPS \
        --num_envs $NUM_ENVS \
        --exp_name "ablation_no_${NAME}" \
        --device $DEVICE \
        --seed 42
    COUNT=$((COUNT + 1))
done

echo -e "\n══════════════════════════════════════════════"
echo "  All ablation experiments complete!"
echo "  Results saved in logs/ablation_*"
echo "══════════════════════════════════════════════"
