"""Fast2D POMDP-like navigation training model. Deliberately NOT Gazebo.

Policy sees only low-dimensional goal direction/history + simulated noisy LiDAR;
training reward may use privileged shortest-path distance, never observation.
"""
from __future__ import annotations
import math
from pathlib import Path
from collections import deque
import numpy as np
from .geometry import lidar,clearance,local_returns
from .filter import apply_filter
from .steering import nominal_steering
from .maps import load_maps,procedural,grid_distance,route_at,ROOM
from .dynamic import join_obstacles, sample_actors
from .scenario import load_gazebo_scenario

N_RAYS=72
OBS_DIM=2*N_RAYS+12
GAMMA=.99

def wrap(a):return math.atan2(math.sin(a),math.cos(a))

def observation(ranges,pose,goal,twist,prev_action,prev_intervention,prev_slack,progress,stall,prev_ranges=None):
    """Exact feature order shared with ROS adapter. Ranges sensor-origin, not center."""
    x,y,yaw=pose
    dx=float(goal[0]-x);dy=float(goal[1]-y)
    err=wrap(math.atan2(dy,dx)-yaw)
    extras=np.array([min(1.,math.hypot(dx,dy)/12.), math.sin(err),math.cos(err),
       twist[0]/.22,twist[1]/2.63,prev_action[0],prev_action[1],
       prev_intervention[0]/.22,prev_intervention[1]/2.63,
       min(prev_slack,2.)/2.,np.clip(progress,-1.,1.),float(stall)],dtype=np.float32)
    ranges=np.nan_to_num(np.asarray(ranges,dtype=np.float32),nan=3.5,posinf=3.5,neginf=3.5)
    prev=np.asarray(ranges if prev_ranges is None else prev_ranges,dtype=np.float32)
    if prev.shape!=ranges.shape:raise ValueError('Previous LiDAR shape mismatch')
    return np.concatenate([np.clip(ranges,.12,3.5)/3.5,np.clip(prev,.12,3.5)/3.5,extras]).astype(np.float32)

class FastNavEnv:
    def __init__(self,map_root='configs',split='train',seed=0,level=2,max_steps=400,
                 mode='full',noise=True,custom_weight=.30,forced_map=None,scenario_path=None):
        assert split in ('train','holdout','dev')
        assert mode in ('full','alpha_only','bias_only','fixed')
        self.rng=np.random.default_rng(seed)
        self.seed=seed;self.root=Path(map_root);self.split=split
        self.level=level;self.max_steps=max_steps;self.mode=mode
        self.noise=noise;self.custom_weight=custom_weight;self.forced_map=forced_map
        if scenario_path is not None and split=='train':raise ValueError('Gazebo benchmark scenarios CANNOT be used for training')
        self.scenario=load_gazebo_scenario(scenario_path) if scenario_path else None
        self.last_scan=np.full(N_RAYS,3.5,dtype=np.float32)
        self.maps=[];self.refresh_maps()
        self.stats={};self.reset()

    def refresh_maps(self):self.maps=load_maps(self.root,self.split)

    def _pick(self):
        if self.scenario:
            q=self.scenario
            return q['obstacles'].copy(),q['start'][:2].copy(),q['goal'].copy(),q['geodesic'],q['name'],q['extent']
        self.refresh_maps() # user can drop new training JSON while running: discovered at next reset
        for trial in range(140):
            if self.split in ('holdout','dev') or self.forced_map or (self.maps and self.rng.random()<self.custom_weight):
                if not self.maps:raise RuntimeError(f'No {self.split} map JSON in {self.root/self.split}')
                m=next((z for z in self.maps if z['name']==self.forced_map),None) if self.forced_map else self.maps[int(self.rng.integers(0,len(self.maps)))];
                if m is None:raise ValueError(f'Unknown map name: {self.forced_map}')
                obs=m['obstacles'].copy()
                p0=np.array(m['starts'][int(self.rng.integers(len(m['starts'])))],dtype=float)
                goal=np.array(m['goals'][int(self.rng.integers(len(m['goals'])))],dtype=float)
                name=m['name'];extent=m['extent']
            else:
                extent=float(self.rng.uniform(4.5,5.6))
                obs=procedural(self.rng,self.level)
                p0=self.rng.uniform(-extent+.9,extent-.9,2)
                goal=self.rng.uniform(-extent+.9,extent-.9,2)
                name=f'procedural_L{self.level}'
                if self.level==4 and self.rng.random()<.65:
                    # U with bottom wall and upper exit. Start inside U, target below bottom.
                    top=float(np.mean(obs[0:2,2]))
                    ox=float((obs[0,1]+obs[1,1])*.5)
                    width=abs(obs[1,1]-obs[0,1])
                    p0=np.array([ox,top+.12])
                    goal=np.array([np.clip(ox+self.rng.uniform(-1,1),-extent+.8,extent-.8),top-2.8])
            if np.linalg.norm(p0-goal)<2.:continue
            if clearance(p0[0],p0[1],obs,extent)<.35 or clearance(goal[0],goal[1],obs,extent)<.38:continue
            dist=grid_distance(obs,goal,extent=extent)
            if dist is None or route_at(dist,p0,extent=extent)>=45.:continue
            return obs,p0,goal,dist,name,extent
        raise RuntimeError('No navigable start-goal pair after 140 attempts; inspect map placement/clearance')

    def reset(self):
        self.obstacles,start,self.goal,self.geodesic,self.map_name,self.extent=self._pick()
        self.pose=np.array([start[0],start[1],self.scenario['start'][2] if self.scenario else self.rng.uniform(-math.pi,math.pi)],dtype=float)
        self.dynamic_actors=(self.scenario['actors'] if self.scenario else sample_actors(self.rng,self.obstacles,self.extent,start,self.goal,self.level))
        self.time_s=0.
        # Limited rotation randomization (not shared with heldout evaluator seeds).
        self.twist=np.zeros(2,dtype=float)
        self.prev_action=np.zeros(2,dtype=float)
        self.prev_intervention=np.zeros(2,dtype=float)
        self.prev_slack=0.;self.prev_progress=0.;self.stall=0
        self.steps=0;self.path_length=0.;self.violation_steps=0;self.max_slack=0.;self.min_clearance=1e6
        self.last_geo=route_at(self.geodesic,self.pose[:2],extent=self.extent)
        self.last_goal_d=float(np.linalg.norm(self.goal-self.pose[:2]))
        self.progress_hist=deque(maxlen=30)
        self.recent_positions=deque(maxlen=30)
        self.last_scan=self._scan()
        self.prev_scan=self.last_scan.copy()
        return self._obs()

    def _scan(self):
        readings=lidar(self.pose[0],self.pose[1],self.pose[2],join_obstacles(self.obstacles,self.dynamic_actors,self.time_s),self.extent,N_RAYS,3.5,-.032)
        if self.noise:
            sig=self.rng.uniform(.001,.014)
            readings=np.clip(readings+self.rng.normal(0.,sig,N_RAYS),.12,3.5)
            drop=self.rng.random(N_RAYS)<self.rng.uniform(0.,.025)
            readings[drop]=3.5
        return np.clip(np.asarray(readings,dtype=np.float32),.12,3.5)

    def _obs(self):
        return observation(self.last_scan,self.pose,self.goal,self.twist,self.prev_action,
                           self.prev_intervention,self.prev_slack,self.prev_progress,self.stall,self.prev_scan)

    def step(self,action):
        action=np.asarray(action,dtype=np.float64)
        expected=2 if self.mode=='full' else (0 if self.mode=='fixed' else 1)
        if action.shape!=(expected,):raise ValueError(f'mode={self.mode} needs action shape ({expected},), got {action.shape}')
        if not np.all(np.isfinite(action)):raise ValueError('Non-finite RL action')
        action=np.clip(action,-1.,1.)
        if self.mode=='full':a,b=float(action[0]),float(action[1])
        elif self.mode=='alpha_only':a,b=float(action[0]),0.
        elif self.mode=='bias_only':a,b=0.,float(action[0])
        else:a,b=0.,0.
        # alpha mapped from normalized [-1,1] to physical [0.2,4.5]
        alpha=3.5 if self.mode in ('fixed','bias_only') else.2+(a+1.)*.5*4.3
        # soft learned steering override; b=0 is original goal heading controller.
        goal_err=wrap(math.atan2(self.goal[1]-self.pose[1],self.goal[0]-self.pose[0])-self.pose[2])
        d_goal=math.hypot(*(self.goal-self.pose[:2]))
        w_nom,steering_gate,goal_cone_range=nominal_steering(
            goal_err,b,self.last_scan,d_goal)
        v_nom=.22*min(1.,d_goal/.35)
        if abs(goal_err)>1.05 and abs(b)<.30:v_nom*=.20
        pts=local_returns(self.last_scan)
        v,w,slack,limit=apply_filter(pts,v_nom,w_nom,alpha)
        # Fast2D ideal twist integration with 4 substeps at 50ms each; NO Gazebo physics equivalence.
        dt=.2;sub_dt=dt/4
        old_pose=self.pose.copy();contact=False
        for j in range(4):
            yaw=self.pose[2];yaw_mid=yaw+w*sub_dt/2.
            self.pose[0]+=v*math.cos(yaw_mid)*sub_dt
            self.pose[1]+=v*math.sin(yaw_mid)*sub_dt
            self.pose[2]=wrap(yaw+w*sub_dt)
            if clearance(self.pose[0],self.pose[1],join_obstacles(self.obstacles,self.dynamic_actors,self.time_s+(j+1)*sub_dt),self.extent)<.16:
                contact=True;break
        self.time_s+=dt
        self.steps+=1
        self.path_length+=float(np.linalg.norm(self.pose[:2]-old_pose[:2]))
        self.twist[:]=[v,w]
        self.prev_action[:]=[a,b]
        self.prev_intervention[:]=[v-v_nom,w-w_nom]
        self.prev_slack=float(slack)
        self.max_slack=max(self.max_slack,slack)
        true_clearance=clearance(self.pose[0],self.pose[1],join_obstacles(self.obstacles,self.dynamic_actors,self.time_s),self.extent)
        self.min_clearance=min(self.min_clearance,true_clearance)
        if true_clearance<.25:self.violation_steps+=1
        new_geod=route_at(self.geodesic,self.pose[:2],extent=self.extent);old_geod=self.last_geo
        self.last_geo=new_geod
        new_goal_d=float(np.linalg.norm(self.goal-self.pose[:2]))
        euclid_progress=self.last_goal_d-new_goal_d
        self.last_goal_d=new_goal_d
        self.progress_hist.append(float(new_goal_d))
        self.recent_positions.append(self.pose[:2].copy())
        moved=(sum(float(np.linalg.norm(self.recent_positions[i]-self.recent_positions[i-1]))
                   for i in range(1,len(self.recent_positions))) if len(self.recent_positions)>1 else 0.)
        self.stall=int(len(self.recent_positions)==self.recent_positions.maxlen and moved<.08 and new_goal_d>.40)
        self.prev_progress=float(np.clip(euclid_progress/.22,-1.,1.))
        goal=new_goal_d<.20
        timeout=self.steps>=self.max_steps
        deadlock=self.stall and len(self.progress_hist)>=30 and self.steps>50
        # Truncate at fixed horizon; stall may end episode early with penalty.
        terminated=bool(goal or contact or deadlock)
        truncated=bool(timeout and not terminated)
        # Potential shaping with gamma=.99: at stationary s, (gamma-1)*Phi(s)<=0.
        pot_prev=max(0.,22.-min(old_geod,22.))*3.
        pot_next=0. if terminated else max(0.,22.-min(new_geod,22.))*3.
        reward=.99*pot_next-pot_prev-.12
        reward-=.8*slack*slack+.14*((v-v_nom)/.22)**2+.06*((w-w_nom)/2.63)**2
        reward-=.025*b*b
        reward-=12.*max(0.,.29-true_clearance)**2
        if goal:reward+=200.
        if contact:reward-=120.
        if deadlock:reward-=30.
        # NOTE no artificial timeout penalty: bootstrapping through time limits.
        self.prev_scan=self.last_scan.copy()
        self.last_scan=self._scan()
        obs=self._obs()
        info={'status':'goal' if goal else 'contact' if contact else 'deadlock' if deadlock else 'timeout' if timeout else 'running',
              'alpha':alpha,'bias':b,'bias_gate':steering_gate,'goal_cone_range_m':goal_cone_range,
              'v':v,'w':w,'w_nom':w_nom,'v_nom':v_nom,'slack':slack,
              'cbf_speed_cap':limit,'clearance_m':true_clearance,'goal_d_m':new_goal_d,
              'geodesic_m':new_geod,'route_progress_m':old_geod-new_geod,
              'path_length_m':self.path_length,'min_clearance_m':self.min_clearance,'violation_steps':self.violation_steps,
              'map_name':self.map_name,'steps':self.steps,'dynamic_actor_count':len(self.dynamic_actors)}
        return obs,float(reward),terminated,truncated,info

    @property
    def action_dim(self):return 2 if self.mode=='full' else 0 if self.mode=='fixed' else 1
