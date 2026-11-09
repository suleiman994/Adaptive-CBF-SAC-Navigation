#!/usr/bin/env bash
# Run one Gazebo Classic world and spawn burger in the scenario's fixed start.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo 'Usage: bash gazebo_v31/run_world.sh S01_open';exit 2; fi
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD="$BASE/worlds/$1.world"
SCENARIO="$BASE/scenarios/$1.json"
[[ -e "$WORLD" && -e "$SCENARIO" ]] || { echo "Unknown scenario $1"; exit 2; }
[[ -f /opt/ros/humble/setup.bash ]] || { echo 'Missing ROS 2 Humble: /opt/ros/humble/setup.bash';exit 2; }
# shellcheck disable=SC1091
set +u
source /opt/ros/humble/setup.bash
set -u
export TURTLEBOT3_MODEL=burger
command -v ros2 >/dev/null || { echo 'ros2 not in PATH';exit 2; }
ros2 pkg prefix gazebo_ros >/dev/null || { echo 'Install gazebo_ros (Gazebo Classic)';exit 2; }
TB3_PREFIX="$(ros2 pkg prefix turtlebot3_gazebo)"
TB3_MODEL="$TB3_PREFIX/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
[[ -f "$TB3_MODEL" ]] || { echo "Missing burger model at $TB3_MODEL";exit 2; }
export GAZEBO_MODEL_PATH="$TB3_PREFIX/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}"
mapfile -t INITIAL < <(python3 - "$SCENARIO" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));print(*s['start'],sep='\n')
PY
)
# Ensure no leftover Gazebo world is running from another experiment.
if pgrep -f '[g]zserver' >/dev/null; then
  echo 'WARNING: another gzserver is running. Stop it before starting a new world.'
  exit 3
fi
ros2 launch gazebo_ros gazebo.launch.py world:="$WORLD" verbose:=false &
GAZEBO_PID=$!
ACTOR_PID=''
cleanup(){
  if [[ -n "$ACTOR_PID" ]]; then kill "$ACTOR_PID" 2>/dev/null || true;fi
  kill "$GAZEBO_PID" 2>/dev/null || true
  wait "$GAZEBO_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
ok=0
for i in $(seq 1 60); do
  if ros2 service list 2>/dev/null | grep -Eq '^(/spawn_entity|/gazebo/spawn_entity)$'; then ok=1;break;fi
  if ! kill -0 "$GAZEBO_PID" 2>/dev/null;then echo 'Gazebo failed to start';exit 4;fi
  sleep 1
done
[[ $ok == 1 ]] || { echo 'Timed out waiting for Gazebo spawn_entity service';exit 4; }
echo 'Spawning TurtleBot3 Burger...'
ros2 run gazebo_ros spawn_entity.py -entity burger -file "$TB3_MODEL" -x "${INITIAL[0]}" -y "${INITIAL[1]}" -z 0.02 -Y "${INITIAL[2]}"
# Dynamic obstacle movement is a separate node and does NOT modify SAC observations.
if python3 - "$SCENARIO" <<'PY'
import json,sys
sys.exit(0 if json.load(open(sys.argv[1]))['dynamic'] else 1)
PY
then
  echo 'Starting scripted obstacle motion...'
  ACTOR_SCRIPT="$BASE/dynamic_actor.py"
  if python3 - "$SCENARIO" <<'PY_YIELD'
import json,sys
s=json.load(open(sys.argv[1]))
sys.exit(0 if s.get('pedestrian_behavior')=='reactive_yield_to_robot' else 1)
PY_YIELD
  then
    ACTOR_SCRIPT="$(dirname "$BASE")/research/s9/dynamic_actor_reactive.py"
  fi
  # S9_LAST2_SLOW_ACTOR_DISPATCH_V1: ONLY for new R08 scene.
  if python3 - "$SCENARIO" <<'PY_SLOW20'
import json,sys
s=json.load(open(sys.argv[1]));sys.exit(0 if s.get('pedestrian_behavior')=='slow_yield_20' else 1)
PY_SLOW20
  then
    ACTOR_SCRIPT="$(dirname "$BASE")/research/s9_last2/slow_crowd_actor.py"
  fi
  # S9_CROWD_V3_INDEPENDENT_DISPATCH: new-scene-only, identical across modes.
  if python3 - "$SCENARIO" <<'PY_INDEPENDENT20'
import json,sys
scene=json.load(open(sys.argv[1]))
sys.exit(0 if scene.get('pedestrian_behavior')=='independent_scripted_20' else 1)
PY_INDEPENDENT20
  then
    ACTOR_SCRIPT="$(dirname "$BASE")/research/s9_crowd_v3/independent_actor.py"
  fi
  python3 "$ACTOR_SCRIPT" --scenario "$SCENARIO" & ACTOR_PID=$!
fi
echo "WORLD READY: $1. In a NEW terminal run gazebo_v31/run_policy.sh ..."
wait "$GAZEBO_PID"
