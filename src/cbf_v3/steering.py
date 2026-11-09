"""V4 hypothesis: LiDAR-conditioned steering override (NOT a safety guarantee).

Nominal path-following still uses bearing to final goal. This test isolates whether
having a bias capable of overturning goal-directed yaw helps escape blocked sectors.
Same function imported by Fast2D and Gazebo adapter. Fixed/alpha-only use b=0.
"""
import math
import numpy as np
W_MAX=2.63
GOAL_CLEAR=1.15
GOAL_BLOCKED=0.50
SECTOR_HALF_WIDTH=math.radians(12.)

def goal_bearing_clearance(scan,goal_error):
    a=np.asarray(scan,dtype=np.float64)
    if a.ndim!=1 or a.size<16:raise ValueError('Need full 360-degree scan')
    r=np.nan_to_num(a,nan=3.5,posinf=3.5,neginf=0.)
    angles=np.arange(a.size)*2.*math.pi/a.size
    diff=np.arctan2(np.sin(angles-goal_error),np.cos(angles-goal_error))
    return float(np.min(r[np.abs(diff)<=SECTOR_HALF_WIDTH]))

def nominal_steering(goal_error,learned_bias,scan,goal_distance):
    w_goal=float(np.clip(1.8*goal_error,-W_MAX,W_MAX))
    r=goal_bearing_clearance(scan,goal_error)
    gate=0. if goal_distance<.35 else float(np.clip((GOAL_CLEAR-r)/(GOAL_CLEAR-GOAL_BLOCKED),0.,1.))
    b=float(np.clip(learned_bias,-1.,1.))
    intensity=gate*abs(b)
    # b=-1 can select a full-rate clockwise turn even if w_goal>0.
    w_nom=float(np.clip((1.-intensity)*w_goal+gate*b*W_MAX,-W_MAX,W_MAX))
    return w_nom,gate,r
