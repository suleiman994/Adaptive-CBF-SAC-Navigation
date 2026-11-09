#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-smoke}"; shift || true
case "$MODE" in
  smoke)
    python3 v4_selfcheck.py
    python3 -m cbf_v3.runner train --mode alpha_only --steps 256 --warmup 32 --batch 32 --envs 4 --gradient-steps 1 --save-interval 256 --log-interval 128 --out outputs_v4_smoke --device cpu
    python3 -m cbf_v3.runner evaluate --mode fixed --split dev --episodes 4 --out outputs_v4_smoke
    ;;
  train)
    FRAMES="${1:-300000}"
    for seed in 41 42 43; do
      for mode in alpha_only bias_only full; do
        echo "TRAIN V4: $mode seed=$seed frames=$FRAMES"
        python3 -m cbf_v3.runner train --mode "$mode" --seed "$seed" --steps "$FRAMES" --envs 16 --maps configs --out outputs_v4_train --device auto
      done
    done
    ;;
  gazebo12_fast2d)
    for seed in 41 42 43; do
      for mode in fixed alpha_only bias_only full; do
        CKPT=()
        if [[ "$mode" != fixed ]]; then CKPT=(--checkpoint "outputs_v4_train/${mode}_seed${seed}/latest.pt"); fi
        python3 -m cbf_v3.runner evaluate --mode "$mode" --split dev --scenario-dir gazebo_v31/scenarios --episodes 12 --seed "$seed" --noise --out "outputs_v4_gazebo12_fast2d_seed${seed}" "${CKPT[@]}"
      done
    done
    ;;
  *) echo "Usage: $0 smoke | train [frames] | gazebo12_fast2d" >&2;exit 2;;
esac
