#!/usr/bin/env bash
# V4 independent Gazebo evaluation; does not use the V3.1 benchmark checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
GROUP="${1:-easy}"; SEED="${2:-42}"
case "$GROUP" in
  easy) SC=(E01_open E02_sparse E03_wide_corridor E04_offset_gate);;
  hard) SC=(H01_narrow_chicane H02_dense_large H03_double_gate H04_maze_large);;
  dynamic) SC=(D01_slow_crossing D02_two_pedestrians D03_slow_crowd D04_mixed_people);;
  *) echo 'Usage: bash run_gazebo_v4.sh easy|hard|dynamic [training_seed]' >&2;exit 2;;
esac
python3 CBF_V4_Gazebo_Benchmark_Runner.py --scenarios "${SC[@]}" --train-seed "$SEED" \
 --output "research/results/GAZEBO_V4_${GROUP}_seed${SEED}" 
