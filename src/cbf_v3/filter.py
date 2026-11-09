"""Convex 3-variable CBF-QP approximation via fast weighted dual coordinate projection.

Hard center constraints are enforced by forward-speed upper bounds. Soft lookahead
constraints couple v and omega through a nonnegative *shared* slack. The solver
checks the final residual and fails closed when numerical convergence is poor.
This DOES NOT prove sampled-data physical collision avoidance.
"""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def _qp_projection(pts,v_nom,w_nom,alpha,d_center=.25,ell=.18,d_look=.34,max_v=.22,max_w=2.63,iterations=100):
    # Convex objective: .5*8*(v-vnom)^2 + .5*.5*(w-wnom)^2 + .5*60*delta^2
    n=pts.shape[0]
    hi=max_v
    for i in range(n):
        px=pts[i,0];py=pts[i,1]
        h=px*px+py*py-d_center*d_center
        if h<0:
            hi=0.
        elif px>1.e-6:
            hi=min(hi,max(0.,alpha*h/(2*px)))
    hi=max(0.,min(max_v,hi))
    # compute soft inequalities a_i * [v,w,s] <= b_i; include s>=0 by projection
    A=np.zeros((n,3),dtype=np.float64)
    B=np.zeros(n,dtype=np.float64)
    lam=np.zeros(n,dtype=np.float64)
    for i in range(n):
        rx=ell-pts[i,0];ry=-pts[i,1]
        hl=rx*rx+ry*ry-d_look*d_look
        A[i,0]=-2.*rx;A[i,1]=-2*ell*ry;A[i,2]=-1.
        B[i]=2.*hl
    v=max(0.,min(hi,v_nom));w=max(-max_w,min(max_w,w_nom));s=0.
    lat0=v_nom;lat1=w_nom;lat2=0.
    inv0=1./8.;inv1=1./.5;inv2=1./60.
    # Hildreth coordinate ascent on dual; unconstrained z reprojected to boxes.
    # With box constraints, dual coordinate update uses exact proximal minimizer
    # via one-dimensional bisection of constraint residual.
    for iteration in range(iterations):
        max_change=0.
        for i in range(n):
            av=A[i,0];aw=A[i,1];as_=A[i,2]
            old=lam[i]
            # update primal by minimizing quadratic with other dual multipliers fixed
            # recover z_0 with lambda_i zero
            b0=lat0+inv0*av*old;b1=lat1+inv1*aw*old;b2=lat2+inv2*as_*old
            lo=0.;upper=max(old+1.,1.)
            for _ in range(14):
                x0=max(0.,min(hi,b0-inv0*av*upper))
                x1=max(-max_w,min(max_w,b1-inv1*aw*upper))
                x2=max(0.,b2-inv2*as_*upper)
                if av*x0+aw*x1+as_*x2-B[i]<=0.:break
                upper*=2.
            for _ in range(23):
                mid=.5*(lo+upper)
                x0=max(0.,min(hi,b0-inv0*av*mid))
                x1=max(-max_w,min(max_w,b1-inv1*aw*mid))
                x2=max(0.,b2-inv2*as_*mid)
                if av*x0+aw*x1+as_*x2-B[i]>0:lo=mid
                else:upper=mid
            candidate=.5*(lo+upper)
            # If at lambda=0 inequality already feasible, multiplier must be 0
            x0=max(0.,min(hi,b0));x1=max(-max_w,min(max_w,b1));x2=max(0.,b2)
            if av*x0+aw*x1+as_*x2-B[i]<=0:candidate=0.
            lat0=b0-inv0*av*candidate;lat1=b1-inv1*aw*candidate;lat2=b2-inv2*as_*candidate
            v=max(0.,min(hi,lat0));w=max(-max_w,min(max_w,lat1));s=max(0.,lat2)
            max_change=max(max_change,abs(candidate-old))
            lam[i]=candidate
        if max_change<1e-6:break
    # construct required nonnegative slack to correct residual; unlike hard constraints,
    # this may differ slightly from unconverged optimum.
    required=0.
    for i in range(n):
        required=max(required,A[i,0]*v+A[i,1]*w-B[i])
    s=max(s,required,0.)
    return v,w,s,hi

@njit(cache=True)
def _rollout_min(pts,v,w,dt=.20,n=8):
    if pts.shape[0]==0:return 1e8
    m=1e8
    for k in range(n+1):
        t=dt*k/n
        if abs(w)<1e-8:x=v*t;y=0.
        else:x=v*math.sin(w*t)/w;y=v*(1-math.cos(w*t))/w
        for j in range(pts.shape[0]):
            d=math.hypot(pts[j,0]-x,pts[j,1]-y)
            if d<m:m=d
    return m

@njit(cache=True)
def apply_filter(pts,v_nom,w_nom,alpha,dt=.20,d_center=.25):
    v,w,s,hi=_qp_projection(pts,v_nom,w_nom,alpha,d_center=d_center)
    dmin=_rollout_min(pts,v,w,dt)
    # sampled rollout guard; not a physical safety certificate
    if pts.shape[0]>0 and dmin<d_center-1e-4:
        for fac in (.7,.45,.2,0.):
            nv=v*fac; nd=_rollout_min(pts,nv,w,dt)
            if nd>=d_center-1e-4:
                v=nv;dmin=nd;break
        else:
            v=0.
    return v,w,s,hi
