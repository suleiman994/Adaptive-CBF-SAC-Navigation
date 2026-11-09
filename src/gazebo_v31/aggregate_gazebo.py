#!/usr/bin/env python3
"""Aggregate only REAL Gazebo episode summary JSONs; never fabricate trial counts."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,json
p=argparse.ArgumentParser();p.add_argument('--root',default='outputs_gazebo_v31');p.add_argument('--write',default='gazebo_results_summary.csv');a=p.parse_args()
groups=defaultdict(list)
for f in Path(a.root).rglob('*_summary.json'):
    d=json.loads(f.read_text());groups[(d['scenario'],d['mode'])].append(d)
rows=[]
for (scene,mode),trials in sorted(groups.items()):
    counts=Counter(r['status'] for r in trials)
    safe_known=[r for r in trials if r['safe_goal'] is not None]
    complete=[r for r in trials if r['ground_truth_complete']]
    rows.append(dict(scenario=scene,mode=mode,n=len(trials),goals=counts['goal'],deadlocks=counts['deadlock'],
                     timeouts=counts['timeout'],goal_success_rate=counts['goal']/len(trials),
                     safe_goal_known_n=len(safe_known),safe_goals=sum(r['safe_goal'] for r in safe_known),
                     safe_goal_rate=(sum(r['safe_goal'] for r in safe_known)/len(safe_known) if safe_known else ''),
                     full_ground_truth_n=len(complete),
                     valid_dynamic_trials=sum(bool(r.get('valid_dynamic_trial',False)) for r in trials),
                     min_clearance_m=(min(r['min_clearance_m'] for r in trials if r['min_clearance_m'] is not None)
                                      if any(r['min_clearance_m'] is not None for r in trials) else '')))
if not rows:raise SystemExit('No Gazebo results found: run at least one real episode')
with open(a.write,'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
for row in rows:print(row)
print('SAVED:',a.write)
