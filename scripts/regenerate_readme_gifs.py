#!/usr/bin/env python3
"""Rebuild README GIFs from frozen step traces and scene geometry included in docs/animation_sources.
This is a telemetry visualization (not screen capture, video, or a safety certificate).
Independent-controller episodes in the four-controller GIF are aligned by elapsed time only.
"""
from pathlib import Path
import csv, json, io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/animation_sources'
OUT=ROOT/'docs/media'
OUT.mkdir(parents=True,exist_ok=True)


def read_csv(filename):
    with (DATA/filename).open(newline='') as f:
        return list(csv.DictReader(f))

def geom(ax, world):
    for ob in world.get('obstacles',[]):
        if ob['type']=='box':
            ax.add_patch(Rectangle((ob['x']-ob['width']/2,ob['y']-ob['height']/2),ob['width'],ob['height'],alpha=.33,edgecolor='black'))
        elif ob['type']=='circle':
            ax.add_patch(Circle((ob['x'],ob['y']),ob['radius'],alpha=.33,edgecolor='black'))
    ax.plot([world['start'][0]],[world['start'][1]],marker='o',markersize=8,label='Start')
    ax.plot([world['goal'][0]],[world['goal'][1]],marker='*',markersize=12,label='Goal')
    if world['name'].startswith('R03') or world['name'].startswith('R06'):
        ax.set_xlim(-5,5)
        ax.set_ylim(-2.3,2.3)
    elif world['name'].startswith('R07'):
        ax.set_xlim(-6,6)
        ax.set_ylim(-5,5)
    else:
        e=world.get('room_half_extent',6)
        ax.set_xlim(-e,e)
        ax.set_ylim(-e,e)
    ax.set_aspect('equal',adjustable='box'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.grid(alpha=.18)


def save_gif(frames,name,ms=220):
    frames[0].save(OUT/name,save_all=True,append_images=frames[1:],duration=ms,loop=0,optimize=True)
    print(f'Created {name}: {(OUT/name).stat().st_size/1024/1024:.2f} MiB')

def fig_image(fig):
    canvas=fig.canvas
    canvas.draw()
    img=np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img).convert('P',palette=Image.Palette.ADAPTIVE,colors=128)


def frame_axes(world,title):
    fig,ax=plt.subplots(figsize=(7.2,4.3),dpi=95)
    geom(ax,world);ax.set_title(title,fontsize=12)
    fig.tight_layout()
    return fig,ax


def rel_time(rows):
    t=np.array([float(r['sim_time_s']) for r in rows])
    return t-t[0]


def run_comparison():
    world=json.loads((DATA/'baffle_world.json').read_text())
    modes=[('fixed','Fixed CBF','Deadlock'),('alpha_only','Gain-only','Deadlock'),('bias_only','Steering-only','Deadlock'),('full','Joint-adaptive','Goal')]
    runs=[(name,label,status,read_csv('baffle_'+name+'_steps.csv')) for name,label,status in modes]
    frames=[]
    T=max(rel_time(r)[-1] for _,_,_,r in runs)
    for t in np.linspace(0,T,38):
        fig,ax=frame_axes(world,'Alternating-baffle corridor — recorded robot trajectories')
        for name,label,status,rows in runs:
            rt=rel_time(rows)
            mask=(rt<=t)
            xs=np.array([float(r['x_odom_world']) for r in rows]); ys=np.array([float(r['y_odom_world']) for r in rows])
            n=max(1,int(mask.sum()))
            line,=ax.plot(xs[:n],ys[:n],lw=2,label=f'{label} ({status})')
            ax.plot(xs[n-1],ys[n-1],marker='o',markersize=5,color=line.get_color())
        ax.text(.02,.04,f'Elapsed time: {t:.1f} s',transform=ax.transAxes,fontsize=10,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        ax.legend(loc='upper left',fontsize=7,ncol=2)
        frames.append(fig_image(fig))
    save_gif(frames,'baffle_four_controller_trajectories.gif')


def run_dynamic(scen,prefix,title,filename):
    world=json.loads((DATA/(prefix+'_world.json')).read_text())
    steps=read_csv(prefix+'_full_steps.csv')
    people=read_csv(prefix+'_full_actor_tracks.csv')
    actors={}
    for row in people:
        actors.setdefault(int(row['step']),[]).append(row)
    xs=np.array([float(r['x_odom_world']) for r in steps]); ys=np.array([float(r['y_odom_world']) for r in steps])
    frames=[]
    for idx in np.linspace(0,len(steps)-1,38).astype(int):
        r=steps[idx]
        fig,ax=frame_axes(world,title)
        line,=ax.plot(xs[:idx+1],ys[:idx+1],lw=2.3,label='Joint-adaptive odometry')
        ax.plot(xs[idx],ys[idx],marker='o',markersize=8,color=line.get_color())
        present=actors.get(int(float(r['step'])),[])
        for j,person in enumerate(present):
            if j==0: lab='Scripted pedestrians (same recorded episode)'
            else:lab=None
            p=next((p for p in world.get('dynamic',[]) if p['name']==person['person']),None)
            radius=p['radius'] if p else .2
            ax.add_patch(Circle((float(person['x_m']),float(person['y_m'])),radius,fill=False,lw=1.5))
            ax.plot(float(person['x_m']),float(person['y_m']),marker='x',markersize=5,label=lab)
        dc=float(r['clearance_m'])
        text=f"Elapsed: {float(r['sim_time_s'])-float(steps[0]['sim_time_s']):.1f} s    Recorded center clearance: {dc:.3f} m"
        if dc<.25: text+='  |  Below 0.25 m margin'
        ax.text(.01,.01,text,transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.88,'edgecolor':'none'})
        ax.legend(loc='upper left',fontsize=7)
        frames.append(fig_image(fig))
    save_gif(frames,filename)

if __name__=='__main__':
    run_comparison()
    run_dynamic('R06','pedestrian','Pedestrian-populated corridor — one Joint-adaptive run','pedestrian_joint_recorded_motion.gif')
    run_dynamic('R07','mixed','Mixed static/dynamic clutter — one Joint-adaptive run','mixed_joint_recorded_motion.gif')
