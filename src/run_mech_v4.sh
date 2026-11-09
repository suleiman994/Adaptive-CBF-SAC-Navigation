#!/usr/bin/env bash
# Run from the project root. Runs 12 real Gazebo worlds × four modes × seeds.
# Usage: bash run_mech_v4.sh 42       # 48 trials, seed-42 pilot
#        bash run_mech_v4.sh all      # 144 trials (41,42,43)
#        bash run_mech_v4.sh report   # regenerate figures from existing trial CSVs
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-42}"
OUT="research/results/MECH_V4_GAZEBO_12"
if [[ "$MODE" == report ]];then
  echo 'Regenerating current reports:'
  python3 research/mech_v4/make_figures.py --run-dir "$OUT"
  exit 0
fi
python3 research/mech_v4/prepare_ros.py
case "$MODE" in
  41|42|43)
    python3 research/mech_v4/run_benchmark.py --seeds "$MODE" --output "$OUT"
    ;;
  all)
    python3 research/mech_v4/run_benchmark.py --seeds 41 42 43 --output "$OUT"
    ;;
  *) echo 'Use: bash run_mech_v4.sh 41|42|43|all|report' >&2;exit 2;;
esac
