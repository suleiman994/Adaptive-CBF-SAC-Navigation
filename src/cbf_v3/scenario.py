"""Parse the SAME JSON geometry used by Gazebo. For evaluation only."""
import json
from pathlib import Path
import numpy as np
from .maps import grid_distance

def load_gazebo_scenario(path):
    x=json.loads(Path(path).read_text())
    if len(x['start'])!=3 or len(x['goal'])!=2:raise ValueError('Invalid start/goal')
    st=[]
    for o in x['obstacles']:
        if o['type']=='box':st.append([0.,o['x'],o['y'],o['width']/2,o['height']/2])
        elif o['type']=='circle':st.append([1.,o['x'],o['y'],o['radius'],0.])
        else:raise ValueError(o['type'])
    sta=np.asarray(st,dtype=np.float64).reshape(-1,5)
    E=float(x['room_half_extent']);goal=np.asarray(x['goal'],dtype=float)
    grid=grid_distance(sta,goal,extent=E)
    if grid is None:raise ValueError('No grid path for '+x['name'])
    return dict(name=x['name'],start=np.array(x['start'],dtype=float),
                goal=goal,obstacles=sta,actors=x.get('dynamic',[]),extent=E,geodesic=grid,
                source=str(path),group=x.get('group','unspecified'))
