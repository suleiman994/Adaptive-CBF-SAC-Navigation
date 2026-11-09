#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash gazebo_v31/run_policy.sh S01_open full [episode_index]';exit 2
fi
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$BASE")"
SCENARIO="$1"; MODE="$2"; EPISODE="${3:-0}"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
ARGS=(--scenario "$SCENARIO" --mode "$MODE" --output "$ROOT/outputs_gazebo_v4" --episode "$EPISODE")
if [[ "$MODE" != fixed ]]; then
  SEED="${CBF_V4_TRAIN_SEED:-42}"
  CP="$ROOT/outputs_v4_train/${MODE}_seed${SEED}/latest.pt"
  [[ -f "$CP" ]] || { echo "Missing checkpoint: $CP"; exit 4; }
  ARGS+=(--checkpoint "$CP")
fi
python3 "$BASE/gazebo_bridge.py" "${ARGS[@]}"
