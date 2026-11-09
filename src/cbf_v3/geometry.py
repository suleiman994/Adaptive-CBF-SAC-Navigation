"""Fast 2D ray casting and robot-disc ground-truth collision. Simulator approximation, not Gazebo."""
import math
import numpy as np
from numba import njit

# obstacle row: [type, center_x, center_y, half_width_or_radius, half_height]
# type 0 axis-aligned box, 1 circle.
@njit(cache=True)
def _box_dist(px,py,cx,cy,hx,hy):
    dx=max(abs(px-cx)-hx,0.0);dy=max(abs(py-cy)-hy,0.0)
    return math.hypot(dx,dy)

@njit(cache=True)
def _ray_box(ox,oy,dx,dy,cx,cy,hx,hy):
    lo=-1e20;hi=1e20
    if abs(dx)<1e-9:
        if ox<cx-hx or ox>cx+hx:return 1e20
    else:
        a=(cx-hx-ox)/dx;b=(cx+hx-ox)/dx
        lo=max(lo,min(a,b));hi=min(hi,max(a,b))
    if abs(dy)<1e-9:
        if oy<cy-hy or oy>cy+hy:return 1e20
    else:
        a=(cy-hy-oy)/dy;b=(cy+hy-oy)/dy
        lo=max(lo,min(a,b));hi=min(hi,max(a,b))
    if hi < max(0.0,lo):return 1e20
    return max(0.0,lo)

@njit(cache=True)
def _ray_circle(ox,oy,dx,dy,cx,cy,r):
    x=ox-cx;y=oy-cy
    b=x*dx+y*dy
    c=x*x+y*y-r*r
    disc=b*b-c
    if disc<0:return 1e20
    q=-b-math.sqrt(disc)
    if q>=0:return q
    q=-b+math.sqrt(disc)
    return max(0.0,q) if q>=0 else 1e20

@njit(cache=True)
def lidar(x,y,heading,obs,half_extent=4.8,n_rays=72,max_range=3.5,scan_offset=-.032):
    out=np.empty(n_rays,dtype=np.float32)
    ox=x+scan_offset*math.cos(heading)
    oy=y+scan_offset*math.sin(heading)
    for i in range(n_rays):
        ang=heading+2*math.pi*i/n_rays
        dx=math.cos(ang);dy=math.sin(ang)
        best=max_range
        # boundary of square room (inside-out intersection)
        if dx>1e-9:best=min(best,(half_extent-ox)/dx)
        if dx< -1e-9:best=min(best,(-half_extent-ox)/dx)
        if dy>1e-9:best=min(best,(half_extent-oy)/dy)
        if dy< -1e-9:best=min(best,(-half_extent-oy)/dy)
        for j in range(obs.shape[0]):
            typ=obs[j,0]
            if typ==0:
                t=_ray_box(ox,oy,dx,dy,obs[j,1],obs[j,2],obs[j,3],obs[j,4])
            else:
                t=_ray_circle(ox,oy,dx,dy,obs[j,1],obs[j,2],obs[j,3])
            if t<best:best=t
        out[i]=max(0.,min(max_range,best))
    return out

@njit(cache=True)
def clearance(x,y,obs,half_extent=4.8):
    d=min(half_extent-abs(x),half_extent-abs(y))
    for j in range(obs.shape[0]):
        if obs[j,0]==0:
            dj=_box_dist(x,y,obs[j,1],obs[j,2],obs[j,3],obs[j,4])
        else:
            dj=max(0.0,math.hypot(x-obs[j,1],y-obs[j,2])-obs[j,3])
        d=min(d,dj)
    return d

@njit(cache=True)
def local_returns(rays,n_points=18,local_radius=1.6,scan_offset=-.032):
    # angular sector minima; scan frame x parallel base x, offset compensated.
    n=rays.shape[0]
    pts=np.zeros((n_points,2),dtype=np.float64)
    count=0
    for sector in range(n_points):
        low=int(sector*n/n_points);high=int((sector+1)*n/n_points)
        best=1e20;bx=0.;by=0.
        for i in range(low,high):
            r=float(rays[i]); ang=2*math.pi*i/n
            xx=r*math.cos(ang)+scan_offset;yy=r*math.sin(ang)
            d=math.hypot(xx,yy)
            if r<3.49 and d<local_radius and d<best:
                best=d;bx=xx;by=yy
        if best<1e10:
            pts[count,0]=bx;pts[count,1]=by
            count+=1
    return pts[:count]
