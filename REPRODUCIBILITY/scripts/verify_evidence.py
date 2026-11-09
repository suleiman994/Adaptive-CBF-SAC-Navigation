#!/usr/bin/env python3
"""Check 8 × 4 V4 evidence counts and logged threshold violations.
Does not establish independent physical collision or a formal safety certificate.
"""
from pathlib import Path
import csv,json,hashlib
root=Path(__file__).resolve().parents[2]
p=root/'REPRODUCIBILITY/data/EPISODE_METRICS_FROZEN_SEED41.csv'
with p.open(newline='') as fh: rows=list(csv.DictReader(fh))
assert len(rows)==32, len(rows)
scenes={r['scenario_id'] for r in rows};modes={r['mode'] for r in rows}
assert scenes=={'R01','R02','R03','R04','R05','R06','R07','R09'}
assert modes=={'fixed','alpha_only','bias_only','full'}
assert len({(r['scenario_id'],r['mode']) for r in rows})==32
for mode,target in {'fixed':(2,2,6),'alpha_only':(2,2,6),'bias_only':(6,6,2),'full':(8,7,0)}.items():
 subset=[r for r in rows if r['mode']==mode]
 v=(sum(int(r['goal']) for r in subset),sum(int(r['safe_goal']) for r in subset),sum(r['status']=='deadlock' for r in subset))
 assert v==target,(mode,v)
 print(mode, 'goal, safe-goal, deadlocks:', v)
r07=next(r for r in rows if r['scenario_id']=='R07' and r['mode']=='full')
assert int(r07['violation_steps'])==6
assert r07['collision_source']=='NOT_MEASURED'
for r in rows:
 src=root/'REPRODUCIBILITY/data/scenarios'/next(p.name for p in (root/'REPRODUCIBILITY/data/scenarios').iterdir() if p.is_dir() and p.name.startswith({'R09':'08'}.get(r['scenario_id'],r['scenario_id'][1:])+'_'))/'DATA'/('STEPS_'+r['mode']+'.csv')
 assert src.exists(),src
 print('',end='')
print('Verified: 32 episode summaries, all plotted source trace files present; no independent contact data.')
print('Episode summary SHA256:',hashlib.sha256(p.read_bytes()).hexdigest())
