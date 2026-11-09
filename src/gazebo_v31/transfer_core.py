"""ROS-independent V4 transfer utilities.

No world geometry, dynamic-actor position or Gazebo ground truth enters policy input.
"""
import json, math
from pathlib import Path
import numpy as np

TWO_PI=2.0*math.pi

def wrap(x):return math.atan2(math.sin(x),math.cos(x))

def yaw_from_quaternion(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

def resample_scan(angle_min,angle_increment,ranges,n_rays=72,max_range=3.5):
    """Nearest real source-ray by circular direction; exact 0,5,...355 degree order.

    Reject partial scans rather than invent 360-degree data. No obstacle interpolation.
    """
    v=np.asarray(ranges,dtype=float)
    if v.ndim!=1 or v.size<180 or angle_increment==0 or not np.isfinite(angle_increment):
        raise ValueError('Expected a full-circle 2D LaserScan (>=180 rays, nonzero increment)')
    span=abs(angle_increment)*(v.size-1)
    if span<2*math.pi-math.radians(3.5) or span>2*math.pi+math.radians(4.):
        raise ValueError(f'Scan covers {math.degrees(span):.2f} degrees, expected ~360')
    angles=np.mod(angle_min+np.arange(v.size)*angle_increment,TWO_PI)
    order=np.argsort(angles);angles=angles[order];v=v[order]
    target=TWO_PI*np.arange(n_rays)/n_rays
    j=np.searchsorted(angles,target,side='left')
    left=(j-1)%v.size;right=j%v.size
    dleft=np.abs((target-angles[left]+math.pi)%TWO_PI-math.pi)
    dright=np.abs((target-angles[right]+math.pi)%TWO_PI-math.pi)
    picked=np.where(dleft<=dright,left,right)
    output=v[picked]
    # Mirror FastNavEnv observations: nonfinite = no-return, clamp physical minima
    output=np.nan_to_num(output,nan=max_range,posinf=max_range,neginf=max_range)
    return np.clip(output,.12,max_range).astype(np.float32)

def odom_to_world(initial_world,initial_odom,current_odom):
    """Anchor odometry at known robot spawn pose; do not read privileged model state."""
    wx,wy,wyaw=initial_world;ox,oy,oyaw=initial_odom;x,y,yaw=current_odom
    rotation=wrap(wyaw-oyaw)
    dx=x-ox;dy=y-oy
    return np.array([wx+math.cos(rotation)*dx-math.sin(rotation)*dy,
                     wy+math.sin(rotation)*dx+math.cos(rotation)*dy,
                     wrap(yaw+rotation)],dtype=float)

def load_scenario(path):
    s=json.loads(Path(path).read_text())
    for k in ('name','start','goal','room_half_extent','obstacles','dynamic'):
        if k not in s:raise ValueError(f'Missing {k} in {path}')
    if len(s['start'])!=3 or len(s['goal'])!=2:raise ValueError('invalid start/goal')
    return s

def static_clearance(p,s):
    x,y=float(p[0]),float(p[1])
    E=float(s['room_half_extent'])
    closest=min(E-abs(x),E-abs(y))
    for o in s['obstacles']:
        if o['type']=='box':
            dx=max(abs(x-o['x'])-o['width']/2,0.)
            dy=max(abs(y-o['y'])-o['height']/2,0.)
            d=math.hypot(dx,dy)
        else:d=max(0.,math.hypot(x-o['x'],y-o['y'])-o['radius'])
        closest=min(closest,d)
    return float(closest)

def full_clearance(p,s,dyn_centers):
    c=static_clearance(p,s)
    if s['dynamic']:
        if not dyn_centers or not all(d['name'] in dyn_centers for d in s['dynamic']):
            return float('nan')
        for d in s['dynamic']:
            x,y=dyn_centers[d['name']]
            c=min(c,max(0.,math.hypot(p[0]-x,p[1]-y)-d['radius']))
    return float(c)

def make_obs(scan,pose,goal,twist,prev_action,prev_intervention,prev_slack,progress,stall,prev_scan=None):
    # Delegate to the original simulator function; no duplicate observation definition.
    from cbf_v3.env import observation
    return observation(scan,pose,goal,twist,prev_action,prev_intervention,prev_slack,progress,stall,prev_scan)

def controller_step(scan,pose,goal,action,mode):
    """Invoke same V4 nominal steering and QP used in Fast2D."""
    from cbf_v3.steering import nominal_steering
    from cbf_v3.filter import apply_filter
    from cbf_v3.geometry import local_returns
    action=np.asarray(action,dtype=np.float64).reshape(-1)
    expected=0 if mode=='fixed' else (2 if mode=='full' else 1)
    if action.size!=expected or not np.all(np.isfinite(action)):
        raise ValueError(f'Unexpected action for {mode}: shape={action.shape}')
    act=np.clip(action,-1.,1.)
    if mode=='full':a,b=float(act[0]),float(act[1])
    elif mode=='alpha_only':a,b=float(act[0]),0.
    elif mode=='bias_only':a,b=0.,float(act[0])
    elif mode=='fixed':a,b=0.,0.
    else:raise ValueError(mode)
    alpha=3.5 if mode in ('fixed','bias_only') else.2+.5*(a+1)*4.3
    e=wrap(math.atan2(goal[1]-pose[1],goal[0]-pose[0])-pose[2])
    d=math.hypot(goal[0]-pose[0],goal[1]-pose[1])
    w_nom,gate,cone=nominal_steering(e,b,scan,d)
    v_nom=.22*min(1.,d/.35)
    if abs(e)>1.05 and abs(b)<.30:v_nom*=.2
    v,w,slack,limit=apply_filter(local_returns(np.asarray(scan,dtype=np.float32)),v_nom,w_nom,alpha)
    return {'v':float(v),'w':float(w),'v_nom':float(v_nom),'w_nom':float(w_nom),
            'alpha':float(alpha),'b':float(b),'a':float(a),'gate':float(gate),
            'goal_sector_range':float(cone),'slack':float(slack),'speed_limit':float(limit),
            'goal_distance':float(d),'goal_error':float(e)}
