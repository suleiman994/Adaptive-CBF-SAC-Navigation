"""Training/evaluation CLI with procedurally generated layouts, GPU SAC, full logging."""
from __future__ import annotations
import argparse,csv,json,os,random,time,hashlib,sys
from collections import Counter,deque
from pathlib import Path
import numpy as np
import torch
from .env import FastNavEnv,OBS_DIM
from .sac import SAC,Replay
from .maps import load_maps

ALL_MODES=['full','alpha_only','bias_only','fixed']

def build_envs(args,split='train',n=None,mode=None,base_seed=None,level=None):
    return [FastNavEnv(map_root=args.maps,split=split,seed=(base_seed if base_seed is not None else args.seed)+97*i,
           level=level if level is not None else 2,max_steps=args.max_steps,mode=mode or args.mode,
           noise=split=='train',custom_weight=args.custom_weight) for i in range(n or args.envs)]

def stage(frac):
    if frac<.10:return 0
    if frac<.24:return 1
    if frac<.46:return 2
    if frac<.65:return 3
    if frac<.83:return 4
    return 5

def serialize_meta(args,updates,elapsed,steps):
    return {'obs_dim':OBS_DIM,'act_dim':2 if args.mode=='full' else 1,'mode':args.mode,
            'seed':args.seed,'frames':steps,'updates':updates,'elapsed_s':elapsed,
            'sim_type':'Fast2D (NOT Gazebo)','maps':str(args.maps),
            'controller_semantics':'v4_temporal_scan_bias_override_a45',
            'notes':'156D temporal LiDAR. No obstacle ground truth to policy; privileged static geodesic only in training reward. V4 scripted movers and V4 steering override.'}

def train(args):
    if args.mode=='fixed':raise SystemExit('fixed baseline needs no training; run evaluate --mode fixed')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():raise SystemExit('Requested CUDA but torch.cuda.is_available() is False')
    if device=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 2))
    run=Path(args.out)/f'{args.mode}_seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
    envs=build_envs(args,level=0);obs=np.stack([e._obs() for e in envs]);rng=np.random.default_rng(args.seed+1337)
    agent=SAC(OBS_DIM,envs[0].action_dim,device=device,hidden=args.hidden,gamma=.99)
    replay=Replay(args.buffer,OBS_DIM,envs[0].action_dim)
    prior=0
    if args.resume:
        meta=agent.restore(args.resume)
        if meta.get('mode')!=args.mode or meta.get('obs_dim')!=OBS_DIM or meta.get('controller_semantics')!='v4_temporal_scan_bias_override_a45':raise ValueError('checkpoint is not V4 compatible; DO NOT resume V3.1 checkpoints')
        prior=int(meta.get('frames',0))
        print('RESUME optimizer weights restored. WARNING: replay buffer and RNG state are NOT restored; continued run is not bitwise reproducible.',flush=True)
    stats=Counter();hist=deque(maxlen=100);reports=[];start=time.perf_counter();prev_stage=-1
    print(f'TRAIN {args.mode} | torch={torch.__version__} device={device} | envs={len(envs)} | action_dim={envs[0].action_dim} | resume_frames={prior}',flush=True)
    frames=prior;last_metrics={}
    while frames < args.steps:
        new_stage=stage(frames/max(1,args.steps))
        if new_stage!=prev_stage:
            for e in envs:e.level=new_stage
            print(f'CURRICULUM level={new_stage} at frames={frames}',flush=True)
            prev_stage=new_stage
        if frames<max(prior,args.warmup):act=rng.uniform(-1.,1.,(len(envs),envs[0].action_dim)).astype(np.float32)
        else:act=agent.act(obs)
        nxt=np.empty_like(obs);rew=np.empty(len(envs),np.float32)
        ter=np.zeros(len(envs),np.float32);infos=[]
        for i,e in enumerate(envs):
            no,r,done,truncated,info=e.step(act[i]);nxt[i]=no
            rew[i]=r;ter[i]=float(done);infos.append(info)
        # Store transition BEFORE resetting a completed episode.
        replay.add_batch(obs,act,rew,nxt,ter)
        for i,e in enumerate(envs):
            if ter[i] or infos[i]['status']=='timeout':
                stats[infos[i]['status']]+=1
                hist.append(float(infos[i]['status']=='goal'))
                with (run/'episodes.csv').open('a',newline='') as f:
                    w=csv.writer(f)
                    if f.tell()==0:w.writerow(['frames','mode','map','status','steps','path_length_m','violations','max_slack','goal_d_m'])
                    w.writerow([frames,args.mode,infos[i]['map_name'],infos[i]['status'],infos[i]['steps'],
                         infos[i]['path_length_m'],infos[i]['violation_steps'],e.max_slack,infos[i]['goal_d_m']])
                nxt[i]=e.reset()
        obs=nxt;frames+=len(envs)
        if frames>=args.warmup and replay.n>=args.batch:
            for _ in range(args.gradient_steps):
                batch=replay.sample(args.batch,device,rng)
                last_metrics=agent.update(batch)
        if frames//args.log_interval != (frames-len(envs))//args.log_interval:
            dur=time.perf_counter()-start
            msg=dict(frames=frames,sec=round(dur,1),fps=round((frames-prior)/max(dur,.001),1),
                     episodes=int(sum(stats.values())),success_last100=round(np.mean(hist),3) if hist else None,
                     current_level=new_stage,**last_metrics)
            print('TRAIN_LOG',json.dumps(msg),flush=True)
            reports.append(msg)
            (run/'train_log.json').write_text(json.dumps(reports,indent=2))
        if frames//args.save_interval!=(frames-len(envs))//args.save_interval:
            meta=serialize_meta(args,agent.updates,time.perf_counter()-start,frames)
            dst=run/'latest.pt';tmp=run/'checkpoint.tmp'
            torch.save(agent.checkpoint(meta),tmp);os.replace(tmp,dst)
            print('CHECKPOINT',dst,'frames',frames,flush=True)
    meta=serialize_meta(args,agent.updates,time.perf_counter()-start,frames)
    dest=run/'latest.pt';torch.save(agent.checkpoint(meta),dest)
    print('DONE',dest,'summary',dict(stats),'elapsed_s',round(time.perf_counter()-start,1),flush=True)


def evaluate(args):
    if args.mode!='fixed' and not args.checkpoint:raise SystemExit('evaluate requires --checkpoint except for fixed')
    if args.split=='holdout' and not load_maps(args.maps,'holdout'):
        raise SystemExit('No holdout maps. Add JSON to configs/holdout.')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    agent=None
    if args.mode!='fixed':
        d=2 if args.mode=='full' else 1
        # Read known-project checkpoint metadata to select NN hidden dimension.
        data=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        mm=data['meta'];hidden=data['actor']['net.0.weight'].shape[0]
        if mm.get('mode')!=args.mode or mm.get('obs_dim')!=OBS_DIM or mm.get('controller_semantics')!='v4_temporal_scan_bias_override_a45':
            raise SystemExit('Checkpoint mode / observation mismatch')
        agent=SAC(OBS_DIM,d,device=device,hidden=hidden)
        agent.restore(args.checkpoint,weights_only=True)
        agent.actor.eval()
    out=Path(args.out)/'evaluation';out.mkdir(parents=True,exist_ok=True)
    scene_files=[]
    if args.scenario_dir:
        # A directory may contain legacy scenarios: never silently mix test cohorts.
        manifest=Path('research/scenarios_12/MANIFEST.csv')
        if not manifest.is_file():raise SystemExit('Missing exact 12-scenario manifest: '+str(manifest))
        with manifest.open(newline='') as f:
            names=[r['name'] for r in csv.DictReader(f)]
        if len(names)!=12 or len(set(names))!=12:raise SystemExit('Manifest is not exactly 12 unique scenarios')
        scene_files=[Path(args.scenario_dir)/(name+'.json') for name in names]
        if any(not p.is_file() for p in scene_files):raise SystemExit('Manifest JSON missing from scenario directory')
    if args.scenario_dir and not scene_files:raise SystemExit('No Gazebo scene JSON in '+args.scenario_dir)
    if scene_files and args.episodes>len(scene_files):raise SystemExit('Requested duplicate scenario episodes. Set --episodes <= number of scenarios; repeated fixed configurations are NOT independent tests.')
    maps=load_maps(args.maps,args.split) if args.split in ('holdout','dev') and not scene_files else []
    rows=[]
    for i in range(args.episodes):
        e=FastNavEnv(map_root=args.maps,split=args.split,seed=args.seed+101*i,level=args.level,
                     mode=args.mode,max_steps=args.max_steps,noise=args.noise,custom_weight=args.custom_weight,
                     forced_map=(maps[i%len(maps)]['name'] if maps else None),
                     scenario_path=(scene_files[i] if scene_files else None))
        obs=e._obs();total=0.
        for k in range(args.max_steps):
            act=np.zeros((0,),np.float32) if args.mode=='fixed' else agent.act(obs,deterministic=True)[0]
            obs,r,done,trunc,info=e.step(act);total+=r
            if done or trunc:break
        rows.append({'split':args.split,'episode':i,'seed':args.seed+101*i,
            'mode':args.mode,'map':e.map_name,'group':(e.scenario['group'] if e.scenario else args.split),'status':info['status'],'steps':info['steps'],
            'path_length_m':info['path_length_m'],'min_clearance_m':info['min_clearance_m'],
            'violation_steps':info['violation_steps'],'max_slack':e.max_slack,'return':total})
    dst=out/f'{"gazebo12" if scene_files else args.split}_{args.mode}_seed{args.seed}.csv'
    with dst.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c=Counter(r['status'] for r in rows)
    print('EVAL_RESULT',json.dumps({'mode':args.mode,'split':args.split,'n':len(rows),'counts':dict(c),
                        'success_rate':c['goal']/len(rows),'csv':str(dst)},indent=2))


def main():
    p=argparse.ArgumentParser(description='Fast2D CBF SAC V3 research toolkit')
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('train','evaluate'):
        q=sub.add_parser(cmd)
        q.add_argument('--mode',choices=ALL_MODES,default='full')
        if cmd=='evaluate':q.add_argument('--scenario-dir',default=None,help='Read exact Gazebo JSON worlds; forbidden in training')
        q.add_argument('--maps',default='configs')
        q.add_argument('--seed',type=int,default=42)
        q.add_argument('--device',default='auto',choices=['auto','cuda','cpu'])
        q.add_argument('--out',default='outputs')
        q.add_argument('--max-steps',type=int,default=400)
        q.add_argument('--custom-weight',type=float,default=.30)
        q.add_argument('--hidden',type=int,default=256)
        if cmd=='train':
            q.add_argument('--steps',type=int,default=500000)
            q.add_argument('--envs',type=int,default=16)
            q.add_argument('--buffer',type=int,default=200000)
            q.add_argument('--batch',type=int,default=512)
            q.add_argument('--warmup',type=int,default=5000)
            q.add_argument('--gradient-steps',type=int,default=2)
            q.add_argument('--log-interval',type=int,default=10000)
            q.add_argument('--save-interval',type=int,default=50000)
            q.add_argument('--resume',default=None)
        else:
            q.add_argument('--split',choices=['train','dev','holdout'],default='dev')
            q.add_argument('--checkpoint')
            q.add_argument('--episodes',type=int,default=100)
            q.add_argument('--noise',action='store_true')
            q.add_argument('--level',type=int,default=5)
    args=p.parse_args()
    if args.command=='train':train(args)
    else:evaluate(args)
if __name__=='__main__':main()
