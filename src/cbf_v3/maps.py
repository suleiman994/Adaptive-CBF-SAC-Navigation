"""V3.1 performance-only replacement for cbf_v3/maps.py.

Keep map generation, JSON handling, return types, diagonal corner rule, and reward
geodesic definition unchanged. Numba compiles the occupancy and Dijkstra loops.
Never deploy while a Python training process is running.
"""
from __future__ import annotations
import heapq, json, math
from pathlib import Path
import numpy as np
from numba import njit
from .geometry import clearance

ROOM=4.8

def read_layout(path):
    obj=json.loads(Path(path).read_text(encoding='utf8'))
    assert obj.get('split') in ('train','dev','holdout'), f'missing split in {path}'
    extent=float(obj.get('room_half_extent', ROOM))
    if not 3.5<=extent<=6.5:raise ValueError('room_half_extent must be 3.5..6.5')
    obstacles=[]
    for item in obj.get('obstacles',[]):
        if item['type']=='box':
            assert item['width']>0 and item['height']>0
            obstacles.append([0.,float(item['x']),float(item['y']),float(item['width'])/2,float(item['height'])/2])
        elif item['type']=='circle':
            assert item['radius']>0
            obstacles.append([1.,float(item['x']),float(item['y']),float(item['radius']),0.])
        else: raise ValueError(f'unsupported obstacle: {item}')
    starts=obj.get('starts',[]);goals=obj.get('goals',[])
    for pts in (starts,goals):
        for p in pts:assert len(p)==2 and max(map(abs,p)) < extent-.2
    return {'name':obj.get('name',Path(path).stem),'split':obj['split'],
            'obstacles':np.array(obstacles,dtype=np.float64).reshape(-1,5),
            'starts':starts,'goals':goals,'extent':extent,'source':str(path)}

def load_maps(root,split):
    out=[]
    for path in sorted((Path(root)/split).glob('*.json')):
        layout=read_layout(path)
        if layout['split'] != split:raise ValueError(f'split mismatch: {path}')
        out.append(layout)
    return out

def procedural(rng,level):
    level=int(level)
    obs=[]
    if level==0:return np.empty((0,5),dtype=np.float64)
    if level in (1,2,5):
        n={1:1,2:5,5:10}[level]
        for _ in range(n):
            x,y=rng.uniform(-3.1,3.1,size=2)
            if rng.random()<.7:obs.append([0.,x,y,rng.uniform(.17,.50),rng.uniform(.17,.55)])
            else:obs.append([1.,x,y,rng.uniform(.18,.52),0.])
    elif level==3:
        gate=rng.uniform(-1.8,1.8);gap=rng.uniform(.65,1.35)
        y0=-3.5;y1=3.5
        a=gate-gap/2.;b=gate+gap/2.
        if a>y0:obs.append([0.,0.,(y0+a)/2.,.13,(a-y0)/2.])
        if y1>b:obs.append([0.,0.,(b+y1)/2.,.13,(y1-b)/2.])
    elif level==4:
        ox=rng.uniform(-1,1);oy=rng.uniform(-1,1)
        width=rng.uniform(1.5,2.6);depth=rng.uniform(1.4,2.4)
        obs.extend([[0.,ox-width/2,oy,.13,depth/2.],
                    [0.,ox+width/2,oy,.13,depth/2.],
                    [0.,ox,oy-depth/2.,width/2.+.13,.13]])
    return np.array(obs,dtype=np.float64).reshape(-1,5)

@njit(cache=True)
def _grid_distance_compiled(obs, goal, radius, cell_requested, extent):
    n=int(round(2.*extent/cell_requested))+1
    cell=2.*extent/(n-1)
    occ=np.empty((n,n),dtype=np.bool_)
    for ix in range(n):
        x=-extent+ix*cell
        for iy in range(n):
            y=-extent+iy*cell
            occ[ix,iy]=clearance(x,y,obs,extent)<radius
    gx=int(min(n-1,max(0,round((goal[0]+extent)/cell))))
    gy=int(min(n-1,max(0,round((goal[1]+extent)/cell))))
    if occ[gx,gy]:return np.empty((0,0),dtype=np.float32)
    dist=np.full((n,n),np.inf,dtype=np.float64)
    dist[gx,gy]=0.
    heap=[(0.,gx,gy)]
    while len(heap)>0:
        d,x,y=heapq.heappop(heap)
        if d>dist[x,y]+1e-5:continue
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0:continue
                xx=x+dx;yy=y+dy
                if 0<=xx<n and 0<=yy<n and not occ[xx,yy]:
                    if dx!=0 and dy!=0 and (occ[x+dx,y] or occ[x,y+dy]):continue
                    ds=math.hypot(dx,dy)*cell
                    nd=d+ds
                    if nd<dist[xx,yy]:
                        dist[xx,yy]=nd;heapq.heappush(heap,(nd,xx,yy))
    return dist.astype(np.float32)

def grid_distance(obs,goal,radius=.29,cell=.20,extent=ROOM):
    d=_grid_distance_compiled(np.ascontiguousarray(obs,dtype=np.float64),
         np.ascontiguousarray(goal,dtype=np.float64),float(radius),float(cell),float(extent))
    return None if d.size==0 else d

def route_at(dist,pos,cell=.20,extent=ROOM):
    if dist is None:return float('inf')
    n=dist.shape[0]
    cell=2*extent/(n-1)
    ix=int(np.clip(np.rint((pos[0]+extent)/cell),0,n-1))
    iy=int(np.clip(np.rint((pos[1]+extent)/cell),0,n-1))
    d=float(dist[ix,iy]);return d if math.isfinite(d) else 50.
