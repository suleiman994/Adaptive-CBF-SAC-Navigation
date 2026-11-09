#!/usr/bin/env python3
"""Run V4 CBF-SAC policies on real Gazebo Classic worlds.

Launch a fresh world/spawn for every (scenario, controller, episode), collect
actual gazebo_bridge.py JSON results and logs, preserve failures. Does not
change controller, training code, scenario geometry, or trained checkpoints.

NOTE: Repeating episode indices does NOT generate new map/start/goal geometry.
Results at the same fixed world are repeated trials, not independent worlds.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCENARIOS = (
    'E01_open','E02_sparse','E03_wide_corridor','E04_offset_gate',
    'H01_narrow_chicane','H02_dense_large','H03_double_gate','H04_maze_large',
    'D01_slow_crossing','D02_two_pedestrians','D03_slow_crowd','D04_mixed_people'
)
MODES = ('fixed', 'alpha_only', 'bias_only', 'full')
FIELDS = ('scenario', 'mode', 'episode', 'status', 'goal', 'safe_goal',
          'steps', 'path_length_m', 'min_clearance_m', 'violation_steps',
          'ground_truth_complete', 'clearance_complete',
          'dynamic_motion_verified', 'collision_source',
          'process_exit_code', 'world_ready', 'result_received',
          'world_log', 'policy_log', 'summary_source',
          'forced_shutdown_after_result', 'policy_timed_out')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                   help='V4 project root; default directory containing this script')
    p.add_argument('--scenarios', nargs='+', default=list(SCENARIOS),
                   help='Scenario names. Default: all 12 worlds')
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--episodes', type=int, default=1,
                   help='Repeated fixed-geometry trials, NOT distinct layouts')
    p.add_argument('--episode-offset', type=int, default=0)
    p.add_argument('--world-timeout', type=float, default=110.)
    p.add_argument('--policy-timeout', type=float, default=200.)
    p.add_argument('--settle', type=float, default=2.)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--train-seed',type=int,default=42,help='V4 trained checkpoint seed; default 42')
    p.add_argument('--fail-fast', action='store_true')
    a = p.parse_args()
    if a.episodes < 1 or a.episode_offset < 0 or a.world_timeout <= 0 or a.policy_timeout <= 0:
        p.error('Invalid episode index/count or timeout')
    return a


def gzserver_exists() -> bool:
    p = subprocess.run(['pgrep', '-x', 'gzserver'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    return p.returncode == 0


def stop_group(proc):
    """Stop only descendants in the process group created by this runner."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGINT, 4.), (signal.SIGTERM, 4.), (signal.SIGKILL, 2.)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
        # Even if parent exited, child processes may still belong to the group.
        # Check the group itself rather than only proc.poll().
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def wait_until_world_ready(proc, path, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if path.exists() and 'WORLD READY:' in path.read_text(errors='replace'):
            return True, 'ready'
        if proc.poll() is not None:
            return False, f'world process exited ({proc.returncode})'
        time.sleep(.35)
    return False, 'world startup timeout'


def wait_for_zombie_gazebo(timeout=15.):
    start = time.monotonic()
    while gzserver_exists() and time.monotonic() - start < timeout:
        time.sleep(.5)
    return not gzserver_exists()


def extract_result(path):
    if not path.exists():
        return None
    candidate = None
    with path.open(errors='replace') as f:
        for line in f:
            if 'GAZEBO_EVAL_RESULT ' in line:
                part = line.split('GAZEBO_EVAL_RESULT ', 1)[1].strip()
                try:
                    candidate = json.loads(part)
                except json.JSONDecodeError:
                    continue
    return candidate


def monitor_policy_until_result(proc, log_path, timeout, result_grace=2.0, poll_interval=0.2):
    """Follow the single authoritative GAZEBO_EVAL_RESULT, not only process exit.

    The bridge writes CSV and summary JSON before logging the result. Some
    rclpy/Gazebo shutdown combinations leave the bridge process alive after
    the result is recorded. Wait a short grace period for clean exit, then
    terminate only the policy process group. Return (code, forced, timed_out).
    """
    deadline = time.monotonic() + timeout
    result_seen_at = None
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc, False, False
        now = time.monotonic()
        if extract_result(log_path) is not None:
            if result_seen_at is None:
                result_seen_at = now
            if now - result_seen_at >= result_grace:
                stop_group(proc)
                return proc.poll(), True, False
        elif now >= deadline:
            stop_group(proc)
            return proc.poll(), False, True
        time.sleep(poll_interval)


def save_manifest(path, rows):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def check_files(root, args):
    if not (root / 'gazebo_v31' / 'gazebo_bridge.py').is_file():
        raise SystemExit(f'gazebo_bridge.py missing under {root}')
    if not (root / 'gazebo_v31' / 'run_world.sh').is_file():
        raise SystemExit(f'run_world.sh missing under {root}')
    for name in args.scenarios:
        for sub, suffix in (('scenarios', '.json'), ('worlds', '.world')):
            p = root / 'gazebo_v31' / sub / f'{name}{suffix}'
            if not p.is_file():
                raise SystemExit(f'Missing {p}')
    for mode in args.modes:
        if mode != 'fixed':
            checkpoint = root / 'outputs_v4_train' / f'{mode}_seed{args.train_seed}' / 'latest.pt'
            if not checkpoint.is_file() and not args.dry_run:
                raise SystemExit(f'Missing checkpoint {checkpoint}')


def run_trial(root, output, scene, mode, episode, args):
    tag = f'{scene}__{mode}__episode{episode:03d}'
    logdir = output / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    world_log = logdir / (tag + '__world.log')
    policy_log = logdir / (tag + '__policy.log')
    row = dict(scenario=scene, mode=mode, episode=episode,
               status='not_started', goal=0, safe_goal='', steps='',
               path_length_m='', min_clearance_m='', violation_steps='',
               ground_truth_complete='', clearance_complete='',
               dynamic_motion_verified='', collision_source='',
               process_exit_code='', world_ready=False, result_received=False,
               world_log=str(world_log), policy_log=str(policy_log), summary_source='',
               forced_shutdown_after_result=False, policy_timed_out=False)
    if gzserver_exists():
        row['status'] = 'blocked_existing_gzserver'
        print('ERROR: gzserver is already running. Stop it before a new paired trial.', flush=True)
        return row
    world = None
    with world_log.open('w') as out:
        try:
            world = subprocess.Popen(['bash', str(root/'gazebo_v31'/'run_world.sh'), scene],
                                     cwd=root, stdout=out, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            ready, reason = wait_until_world_ready(world, world_log, args.world_timeout)
            if not ready:
                row['status'] = reason.replace(' ', '_')
                print(f'WORLD ERROR {tag}: {reason}; see {world_log}', flush=True)
                return row
            row['world_ready'] = True
            time.sleep(args.settle)
            cp = root/'outputs_v4_train'/f'{mode}_seed{args.train_seed}'/'latest.pt'
            cmd = ['python3', str(root/'gazebo_v31'/'gazebo_bridge.py'),
                   '--scenario', scene, '--mode', mode, '--episode', str(episode),
                   '--output', str(output/'episodes'), '--device', 'cpu']
            # S9_SCENE_HORIZON_V2: explicit per-scene evaluation time budget.
            horizon = json.loads((root/'gazebo_v31/scenarios'/f'{scene}.json').read_text()).get('max_steps',400)
            cmd.extend(['--max-steps',str(int(horizon))])
            if mode != 'fixed':
                cmd.extend(['--checkpoint', str(cp)])
            # Source ROS 2 outside of set -u. No Gazebo process shares this group.
            bash_cmd = ['bash', '-c', 'source /opt/ros/humble/setup.bash; exec "$@"', 'ros_wrapper'] + cmd
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            policy = None
            with policy_log.open('w') as fp:
                try:
                    policy = subprocess.Popen(bash_cmd, cwd=root, env=env,
                                              stdout=fp, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    code, forced, timed_out = monitor_policy_until_result(
                        policy, policy_log, timeout=args.policy_timeout)
                    row['process_exit_code'] = code
                    row['forced_shutdown_after_result'] = forced
                    row['policy_timed_out'] = timed_out
                    if forced:
                        print(f'INFO {tag}: result saved; lingering ROS process terminated ' 
                              'after 2s grace.', flush=True)
                    if timed_out:
                        row['status'] = 'policy_timeout'
                        print(f'POLICY TIMEOUT {tag}; see {policy_log}', flush=True)
                finally:
                    if policy is not None and policy.poll() is None:
                        stop_group(policy)
            result = extract_result(policy_log)
            if result is not None:
                row['result_received'] = True
                row['summary_source'] = result.get('source_world', '')
                for key in ('status', 'goal', 'safe_goal', 'steps', 'path_length_m',
                            'min_clearance_m', 'violation_steps', 'ground_truth_complete',
                            'clearance_complete', 'dynamic_motion_verified', 'collision_source'):
                    row[key] = result.get(key, '')
                if row['process_exit_code'] not in (0, '') and not row['forced_shutdown_after_result']:
                    print(f'WARNING {tag}: result produced but process exit={row["process_exit_code"]}', flush=True)
            elif row['status'] == 'not_started':
                row['status'] = 'policy_failed_no_result'
            print(f'RESULT {tag}: status={row["status"]} goal={row["goal"]} '
                  f'safe_goal={row["safe_goal"]} clearance={row["min_clearance_m"]}', flush=True)
            return row
        finally:
            stop_group(world)
            if not wait_for_zombie_gazebo():
                print('WARNING: lingering gzserver found; will not kill unrelated processes.', flush=True)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    check_files(root, args)
    dt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output.expanduser().resolve() if args.output else root/'gazebo_benchmarks'/dt)
    trials = [(s, m, e) for s in args.scenarios for m in args.modes
              for e in range(args.episode_offset, args.episode_offset + args.episodes)]
    print(f'ROOT {root}\nOUTPUT {output}\nPLANNED TRIALS {len(trials)}', flush=True)
    print('IMPORTANT: episode indices are filenames only, NOT randomized geometry.', flush=True)
    if args.dry_run:
        for s, m, e in trials:
            print(f'DRY-RUN {s} {m} episode={e}')
        return 0
    if not Path('/opt/ros/humble/setup.bash').exists():
        raise SystemExit('ROS Humble setup.bash is missing')
    if gzserver_exists():
        raise SystemExit('Another Gazebo server is running. Stop it before starting batch.')
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(root=str(root), scenarios=args.scenarios, modes=args.modes,
               episodes=args.episodes, episode_offset=args.episode_offset,
               world_timeout=args.world_timeout, policy_timeout=args.policy_timeout,
               generated_utc=dt, geometry_randomization=False,
               checkpoint_seed=args.train_seed,
               policy_semantics='v4_temporal_scan_bias_override_a45',
               note='One episode per fixed world/start/goal by default; not independent map geometries.')
    (output/'protocol.json').write_text(json.dumps(cfg, indent=2)+'\n')
    rows = []
    csv_path = output/'benchmark_summary.csv'
    try:
        for i, (scene, mode, episode) in enumerate(trials, 1):
            print(f'\n[{i}/{len(trials)}] {scene} {mode} episode={episode}', flush=True)
            row = run_trial(root, output, scene, mode, episode, args)
            rows.append(row)
            save_manifest(csv_path, rows)
            if args.fail_fast and (not row['result_received'] or
                                   (row['process_exit_code'] not in (0, '') and
                                    not row['forced_shutdown_after_result'])):
                print('FAIL FAST: stopping after abnormal execution', flush=True)
                break
    except KeyboardInterrupt:
        print('INTERRUPTED BY USER: partial results already written.', flush=True)
    print('\nSAVED SUMMARY:', csv_path, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
