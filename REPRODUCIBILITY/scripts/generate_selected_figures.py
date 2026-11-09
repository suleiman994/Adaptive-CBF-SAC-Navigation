#!/usr/bin/env python3
"""Regenerate core publication figures from the included frozen Seed-41
CSV files and scene JSON. Does NOT simulate/retrain and does NOT consume
Mohammad--Bezzo development-only results. Requires matplotlib, pandas, numpy.
Usage: python3 REPRODUCIBILITY/scripts/generate_selected_figures.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
DAT = ROOT/'REPRODUCIBILITY/data'
OUT = ROOT/'PAPER_MAIN/figures'
SUPP = ROOT/'SUPPLEMENTARY/figures_by_scenario'
OUT.mkdir(parents=True,exist_ok=True)
order=['R01','R02','R03','R04','R05','R06','R07','R09']
modes=['fixed','alpha_only','bias_only','full']
mode_names={'fixed':'Fixed CBF','alpha_only':r'$\alpha$ only','bias_only':'Bias only','full':'Full adaptive'}
scen_dirs={p.name.split('_')[0].replace('08','R09').replace('07','R07').replace('06','R06').replace('05','R05').replace('04','R04').replace('03','R03').replace('02','R02').replace('01','R01'):p for p in (DAT/'scenarios').iterdir() if p.is_dir()}
# deterministic explicit mapping: suffix names are human-friendly and match R01 ... R07,R09
scen_dirs = {sid:next(p for p in (DAT/'scenarios').iterdir() if p.name.startswith(f'{idx:02d}_')) for idx,sid in enumerate(order,1)}
metrics=pd.read_csv(DAT/'EPISODE_METRICS_FROZEN_SEED41.csv')
assert len(metrics)==32
assert set(metrics['scenario_id'])==set(order)
assert set(metrics['mode'])==set(modes)
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})

def scene(sid):
 return json.loads((scen_dirs[sid]/'DATA/SCENE_GEOMETRY_AND_SOURCE.json').read_text())['scene_definition']
def trace(sid, mode):
 return pd.read_csv(scen_dirs[sid]/f'DATA/STEPS_{mode}.csv')
def scene_draw(ax,sid,legend=False):
 s=scene(sid)
 for o in s.get('obstacles',[]):
  if o['type']=='box':ax.add_patch(Rectangle((o['x']-o['width']/2,o['y']-o['height']/2),o['width'],o['height'],fc='#606b77',ec='#414c56',lw=.7,zorder=3))
  elif o['type']=='circle':ax.add_patch(Circle((o['x'],o['y']),o['radius'],fc='#606b77',ec='#414c56',lw=.7,zorder=3))
 for o in s.get('dynamic',[]):
  ax.add_patch(Circle((o['x'],o['y']),o.get('radius',.2),fill=False,ec='#b07646',ls='--',lw=1,zorder=3))
 ax.plot([s['start'][0]],[s['start'][1]],marker='o',ms=6,color='#1b8c74',zorder=6)
 ax.plot([s['goal'][0]],[s['goal'][1]],marker='*',ms=10,color='#bf5649',zorder=6)
 he=s.get('room_half_extent',5.2)
 ax.set(xlim=(-he,he),ylim=(-he,he));ax.set_aspect('equal',adjustable='box')
 ax.grid(alpha=.2,linewidth=.4)

# Scenario atlas from source geometries (not screenshots)
fig,axs=plt.subplots(2,4,figsize=(12.4,6.1),constrained_layout=True)
for i,(sid,ax) in enumerate(zip(order,axs.ravel())):
 scene_draw(ax,sid)
 ax.set_title(f'({chr(97+i)}) {sid} · {scene(sid).get("group","static")}',fontsize=9)
 if i//4==1:ax.set_xlabel('x (m)')
 if i%4==0:ax.set_ylabel('y (m)')
fig.savefig(OUT/'Fig03_Eight_Frozen_Scenarios_REGENERATED.pdf',bbox_inches='tight')
plt.close(fig)

# Single figure: paired outcomes and per-method counts
fig,(ax,ay)=plt.subplots(1,2,figsize=(11.8,4.0),gridspec_kw={'width_ratios':[1.7,1.0]},constrained_layout=True)
vals=np.zeros((len(order),len(modes)),dtype=int)
letters=[]
for i,sid in enumerate(order):
 ll=[]
 for j,mode in enumerate(modes):
  row=metrics[(metrics['scenario_id']==sid)&(metrics['mode']==mode)].iloc[0]
  val=2 if row.goal and row.safe_goal else (1 if row.goal else 0)
  vals[i,j]=val
  ll.append('S' if val==2 else ('G*' if val==1 else 'D'))
 letters.append(ll)
ax.imshow(vals,aspect='auto',cmap=ListedColormap(['#e7e7e7','#e6bca9','#b7d8cf']),vmin=-.5,vmax=2.5)
ax.set_yticks(range(8),order)
ax.set_xticks(range(4),['Fixed','Alpha','Bias','Full'])
for i in range(8):
 for j in range(4):ax.text(j,i,letters[i][j],va='center',ha='center',fontweight='bold',fontsize=11)
ax.set_title('(a) Paired scenario outcomes (Seed 41)')
ax.set_xlabel('D = deadlock; S = threshold-safe goal; G* = goal with violation')
g=[int(metrics[metrics['mode']==m]['goal'].sum()) for m in modes]
s=[int(metrics[metrics['mode']==m]['safe_goal'].sum()) for m in modes]
x=np.arange(4)
ay.bar(x-.17,g,.34,label='Goal',color='#2e7f9d')
ay.bar(x+.17,s,.34,label='Threshold-safe goal',color='#92bca9')
ay.set_xticks(x,['Fixed','Alpha','Bias','Full'])
ay.set(ylim=(0,8.8),ylabel='Scenarios / 8')
for j in range(4):
 ay.text(j-.17,g[j]+.1,str(g[j]),ha='center',fontsize=8)
 ay.text(j+.17,s[j]+.1,str(s[j]),ha='center',fontsize=8)
ay.set_title('(b) Count summary (one run/layout)')
ay.legend(frameon=False,fontsize=8,loc='upper left')
ay.spines[['top','right']].set_visible(False)
fig.savefig(OUT/'Fig04_Frozen_Ablation_Summary.pdf',bbox_inches='tight')
fig.savefig(OUT/'Fig04_Frozen_Ablation_Summary.png',dpi=220,bbox_inches='tight')
plt.close(fig)

# Four-controller trajectory comparisons, identical axes and map per scene
for sid in ['R03','R06']:
 fig,axs=plt.subplots(1,4,figsize=(12.0,3.1),constrained_layout=True)
 for j,mode in enumerate(modes):
  ax=axs[j];scene_draw(ax,sid)
  t=trace(sid,mode); r=metrics[(metrics.scenario_id==sid)&(metrics['mode']==mode)].iloc[0]
  ax.plot(t.x_odom_world,t.y_odom_world,color='#205a9b',lw=1.5,zorder=4)
  ax.plot([t.x_odom_world.iloc[-1]],[t.y_odom_world.iloc[-1]],marker='x',ms=7,mew=1.6,color='#9b4c40',zorder=7)
  ax.set_title(f'{mode_names[mode]}\n{r.status}, {int(r.steps)} steps',fontsize=9)
  ax.set_xlabel('x (m)')
  if j==0:ax.set_ylabel('y (m)')
 target = OUT if sid == 'R03' else SUPP/'06_pedestrians_in_a_corridor'
 target.mkdir(parents=True,exist_ok=True)
 fig.savefig(target/f'Fig05_{sid}_Paired_Trajectories.pdf',bbox_inches='tight')
 plt.close(fig)

# State-conditioned maps from pooled full trajectory logs
pool=pd.concat([trace(s,'full').assign(scenario=s) for s in order],ignore_index=True)
pool=pool[np.isfinite(pool['clearance_m']) & np.isfinite(pool['heading_error_rad'])]
x=pool['heading_error_rad'].to_numpy();y=pool['clearance_m'].to_numpy()
xb=np.linspace(-np.pi,np.pi,19);yb=np.linspace(0,2.0,17)
num=np.histogram2d(y,x,bins=[yb,xb])[0]
fig,axs=plt.subplots(1,3,figsize=(11.8,3.35),constrained_layout=True)
for j,(column,title) in enumerate([('alpha',r'(a) Adaptive gain $\alpha_k$'),('b',r'(b) Steering output $b_k$'),('gate',r'(c) Steering gate $g_k$')]):
 v=pool[column].to_numpy();tot=np.histogram2d(y,x,bins=[yb,xb],weights=v)[0]
 with np.errstate(invalid='ignore',divide='ignore'):value=np.where(num>=8,tot/num,np.nan)
 im=axs[j].pcolormesh(xb,yb,value,cmap='viridis',rasterized=True)
 axs[j].set_title(title)
 axs[j].set(xlabel='Heading error (rad)',ylabel='Logged center distance (m)')
 fig.colorbar(im,ax=axs[j],pad=.02,shrink=.8)
fig.savefig(OUT/'Fig06_Adaptive_Maps_Reproduced.pdf',bbox_inches='tight')
plt.close(fig)

# R09 executed velocity and correction. Use recorded sim time, not steps*Ts.
t=trace('R09','full');tm=t.sim_time_s.to_numpy()-float(t.sim_time_s.iloc[0])
fig,axs=plt.subplots(1,3,figsize=(11.8,3.2),constrained_layout=True)
axs[0].plot(tm,t.v_nom,label='Nominal $v$',lw=1.1)
axs[0].plot(tm,t.v_cmd,label='Command $v$',lw=1)
axs[0].plot(tm,t.v_odom,label='Odom $v$',lw=.8,alpha=.7)
axs[0].set(ylabel='Linear velocity (m/s)',xlabel='Relative sim time (s)',title='(a) R09: linear command')
axs[0].legend(frameon=False,fontsize=7)
axs[1].plot(tm,t.w_nom,label=r'Nominal $\omega$',lw=1)
axs[1].plot(tm,t.w_cmd,label=r'Command $\omega$',lw=1)
axs[1].plot(tm,t.w_odom,label=r'Odom $\omega$',lw=.8,alpha=.7)
axs[1].set(ylabel='Angular velocity (rad/s)',xlabel='Relative sim time (s)',title='(b) R09: angular command')
axs[1].legend(frameon=False,fontsize=7)
f=[float(metrics[(metrics.scenario_id==sid)&(metrics['mode']=='full')]['frac_nonzero_v_filter_delta'].iloc[0]) for sid in order]
axs[2].bar(range(8),np.array(f)*100,color='#719bb3')
axs[2].set_xticks(range(8),order,rotation=60)
axs[2].set(ylim=(0,100),ylabel='Recorded steps (%)',title='(c) Full: nominal-command correction')
for ax in axs: ax.grid(alpha=.2)
fig.savefig(OUT/'Fig07_Executed_Velocities_and_Intervention.pdf',bbox_inches='tight')
plt.close(fig)

# R07 failure: observed clearance and violation locations
t=trace('R07','full');tm=t.sim_time_s.to_numpy()-float(t.sim_time_s.iloc[0]);bad=t.clearance_m.to_numpy()<.25
fig,(ax,ay)=plt.subplots(1,2,figsize=(11.8,3.7),constrained_layout=True)
scene_draw(ax,'R07');ax.plot(t.x_odom_world,t.y_odom_world,color='#205a9b',lw=1.35)
ax.scatter(t.x_odom_world[bad],t.y_odom_world[bad],s=10,color='#b94a48',zorder=7)
ax.set(xlabel='x (m)',ylabel='y (m)',title='(a) Full: trajectory and sub-threshold samples')
ay.plot(tm,t.clearance_m,lw=1.25,color='#205a9b',label='Logged center distance')
ay.axhline(.25,ls='--',lw=1.2,color='#c15c3c',label='0.25 m threshold')
ay.scatter(tm[bad],t.clearance_m.to_numpy()[bad],s=15,color='#b94a48',zorder=6)
ay.set(xlabel='Relative sim time (s)',ylabel='Center-to-obstacle distance (m)',title=f'(b) {int(bad.sum())} recorded sub-threshold samples')
ay.legend(frameon=False,fontsize=8)
fig.savefig(OUT/'Fig08_R07_Safety_Limitations_REPRODUCED.pdf',bbox_inches='tight')
plt.close(fig)
print('Figures generated in',OUT)
