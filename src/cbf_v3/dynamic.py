"""Fast2D scripted moving circles, identical sinusoid convention to gazebo_v31/dynamic_actor.py.

Ground truth is for simulation geometry, collision metric and reward ONLY;
never provided directly to the policy, steering or QP. Their current planar
surfaces *are* visible to ray casting. No CBF guarantee for moving obstacles.
"""
import math
import numpy as np
from .geometry import clearance

def positions(actors,t):
    rows=[]
    for o in actors:
        phase=2.*math.pi*t/o['period_s']+o.get('phase',0.)
        dx=o['amplitude']*math.sin(phase) if o['axis']=='x' else 0.
        dy=o['amplitude']*math.sin(phase) if o['axis']=='y' else 0.
        rows.append([1.,o['x']+dx,o['y']+dy,o['radius'],0.])
    return np.asarray(rows,dtype=np.float64).reshape(-1,5)

def join_obstacles(static,actors,t):
    if not actors:return static
    return np.concatenate([static,positions(actors,t)],axis=0)

def sample_actors(rng,static,extent,start,goal,level):
    """Disjoint procedural moving actors for TRAIN, not test scenarios."""
    if level<2 or rng.random()>.60:return []
    nmax=1 if level<=2 else 2 if level<=3 else 4
    desired=int(rng.integers(1,nmax+1));out=[]
    for _ in range(desired):
        for _ in range(50):
            axis='x' if rng.random()<.5 else 'y'
            amp=float(rng.uniform(.28,.72))
            period=float(rng.uniform(32.,70.)) # max speed <= 0.1414m/s
            x,y=rng.uniform(-extent+1.3,extent-1.3,2)
            actor=dict(x=float(x),y=float(y),axis=axis,amplitude=amp,
                       period_s=period,phase=float(rng.uniform(-math.pi,math.pi)),radius=.20)
            valid=True
            for k in range(16):
                a=positions([actor],period*k/16.)[0]
                p=a[1:3]
                if clearance(p[0],p[1],static,extent)<.34 or np.linalg.norm(p-start)<.6 or np.linalg.norm(p-goal)<.5:
                    valid=False;break
                for other in out:
                    q=positions([other],period*k/16.)[0,1:3]
                    if np.linalg.norm(p-q)<.52:valid=False;break
                if not valid:break
            if valid:
                out.append(actor);break
    return out
