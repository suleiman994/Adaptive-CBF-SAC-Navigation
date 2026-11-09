#!/usr/bin/env python3
"""Single-episode ROS2/Humble bridge for the V4 checkpoint (independent from frozen V3.1).

- Policy input: 72 LiDAR rays, goal relative to odometry, twist, previous controls.
- Gazebo model_states: EVALUATION ONLY, never passed to the actor/nominal CBF.
- No guarantees of collision-free navigation. Logs experimental clearances.
"""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from collections import deque
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from transfer_core import (load_scenario,resample_scan,yaw_from_quaternion,odom_to_world,wrap,
                            make_obs,controller_step,full_clearance,static_clearance)

def args_cli():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',required=True,help='e.g. S01_open or path to JSON')
    p.add_argument('--mode',choices=['fixed','alpha_only','bias_only','full'],required=True)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',default=str(ROOT/'outputs_gazebo_v31'))
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=400)
    p.add_argument('--control-period',type=float,default=.20)
    p.add_argument('--scan-topic',default='/scan')
    p.add_argument('--odom-topic',default='/odom')
    p.add_argument('--cmd-topic',default='/cmd_vel')
    a=p.parse_args()
    if a.mode!='fixed' and not a.checkpoint:p.error('Non-fixed modes require --checkpoint')
    if a.control_period!=.20:p.error('FROZEN checkpoint requires --control-period 0.20')
    return a

def load_agent(a):
    if a.mode=='fixed':return None
    import torch
    from cbf_v3.sac import SAC
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=data['meta']
    if m.get('mode')!=a.mode or m.get('obs_dim')!=156 or m.get('controller_semantics')!='v4_temporal_scan_bias_override_a45':
        raise RuntimeError('Refusing wrong policy checkpoint / observation schema')
    hidden=int(data['actor']['net.0.weight'].shape[0])
    agent=SAC(156,2 if a.mode=='full' else 1,device=a.device,hidden=hidden)
    agent.restore(a.checkpoint,weights_only=True)
    agent.actor.eval()
    return agent


def main():
    a=args_cli()
    sc_path=Path(a.scenario)
    if not sc_path.is_file():sc_path=HERE/'scenarios'/f'{a.scenario}.json'
    s=load_scenario(sc_path)
    agent=load_agent(a)
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from gazebo_msgs.msg import ModelStates
    from std_msgs.msg import Bool,String

    class Bridge(Node):
        def __init__(self):
            super().__init__('cbf_sac_v31_gazebo_bridge')
            if self.has_parameter('use_sim_time'):
                self.set_parameters([
                    Parameter('use_sim_time', value=True)
                ])
            else:
                self.declare_parameter('use_sim_time', True)
            self.s=s;self.a=a;self.agent=agent;self.scan=None;self.prev_scan=None;self.odom=None
            self.scan_received=0.;self.odom_received=0.
            self.initial_odom=None;self.gt_poses={};self.gt_received=0.
            self.prev_action=np.zeros(2,dtype=np.float32)
            self.prev_intervention=np.zeros(2,dtype=np.float32)
            self.prev_slack=0.;self.progress=0.;self.stall=0
            self.prev_goal_distance=None;self.recent=deque(maxlen=30)
            self.steps=0;self.distance=0.;self.prev_pose=None
            self.violation_steps=0;self.min_clearance=float('inf');self.violated=False
            self.done=False;self.stop_time=None;self.rows=[];self.start_wall=time.monotonic()
            self.actors_active=False
            self.actor_positions={d["name"]:[] for d in s["dynamic"]}
            # S9_ACTOR_TELEMETRY_V2: ground-truth positions logged, never policy input.
            self.actor_tracks=[]
            self.actor_start_pub=self.create_publisher(String,'/cbf_v31/episode_start',10)
            self.create_subscription(Bool,'/cbf_v31/actors_active',self.on_actors_active,10)
            self.pub=self.create_publisher(Twist,a.cmd_topic,10)
            self.create_subscription(LaserScan,a.scan_topic,self.on_scan,qos_profile_sensor_data)
            self.create_subscription(Odometry,a.odom_topic,self.on_odom,qos_profile_sensor_data)
            self.create_subscription(ModelStates,'/model_states',self.on_gt,10)
            self.create_subscription(ModelStates,'/gazebo/model_states',self.on_gt,10)
            self.timer=self.create_timer(a.control_period,self.tick)
            self.get_logger().info(f'V4 {a.mode} | {s["name"]} | goal={s["goal"]} | start={s["start"]} | 5Hz')

        def on_actors_active(self,msg):
            if msg.data:self.actors_active=True

        def on_scan(self,msg):
            try:self.scan=resample_scan(msg.angle_min,msg.angle_increment,msg.ranges)
            except ValueError as exc:
                self.get_logger().error(f'Invalid scan: {exc}');self.scan=None
                return
            self.scan_received=time.monotonic()

        def on_odom(self,msg):
            self.odom=np.array([msg.pose.pose.position.x,msg.pose.pose.position.y,
                                yaw_from_quaternion(msg.pose.pose.orientation)],dtype=float)
            self.twist=np.array([msg.twist.twist.linear.x,msg.twist.twist.angular.z],dtype=float)
            self.odom_received=time.monotonic()
            if self.initial_odom is None:self.initial_odom=self.odom.copy()

        def on_gt(self,msg):
            self.gt_poses={name:(p.position.x,p.position.y) for name,p in zip(msg.name,msg.pose)}
            self.gt_received=time.monotonic()

        def publish(self,v,w):
            cmd=Twist();cmd.linear.x=float(v);cmd.angular.z=float(w);self.pub.publish(cmd)

        def finish(self,status):
            if self.done:return
            self.done=True;self.publish(0,0)
            output=Path(a.output)/s['name'];output.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            prefix=f'{a.mode}_episode{a.episode:03d}_{stamp}'
            csvpath=output/(prefix+'_steps.csv');summary_path=output/(prefix+'_summary.json')
            if self.rows:
                with csvpath.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
            # S9_ACTOR_TELEMETRY_V2 CSV, precise simulator timestamps.
            track_file=output/(prefix+'_actor_tracks.csv')
            if self.actor_tracks:
                with track_file.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(self.actor_tracks[0]))
                    writer.writeheader();writer.writerows(self.actor_tracks)
            c=self.min_clearance
            # strict comparison only when full true clearance is available in all steps
            gt_complete=bool(self.rows) and all(r['clearance_source']=='gazebo_model_states' for r in self.rows)
            clearance_complete=bool(self.rows) and all(math.isfinite(r['clearance_m']) for r in self.rows)
            # S9_DYNAMIC_AUDIT_V2: movement verification, not model safety or contact.
            actor_displacement_m={name:(max(math.hypot(x-pts[0][0],y-pts[0][1])
                for x,y in pts) if len(pts)>1 else 0.0)
                for name,pts in self.actor_positions.items()}
            actor_frame_coverage=all(len(pts)>1 for pts in self.actor_positions.values())
            actor_move_count=sum(d>.04 for d in actor_displacement_m.values())
            dyn_valid=(not s['dynamic'] or (actor_frame_coverage and
                actor_move_count>=max(1,math.ceil(.5*len(s['dynamic']))) and
                max(actor_displacement_m.values(),default=0.)>.10))
            summary=dict(scenario=s['name'],mode=a.mode,episode=a.episode,status=status,steps=self.steps,
                         goal=int(status=='goal'),path_length_m=self.distance,
                         min_clearance_m=c if math.isfinite(c) else None,
                         violation_steps=self.violation_steps,
                         safe_goal=(int(status=='goal' and self.violation_steps==0) if gt_complete and clearance_complete else None),
                         ground_truth_complete=gt_complete,clearance_complete=clearance_complete,
                         dynamic_motion_verified=dyn_valid,valid_dynamic_trial=dyn_valid,
                         actor_tracks_file=str(track_file) if self.actor_tracks else None,
                         actor_displacement_m=actor_displacement_m,
                         actor_motion_verification_method='S9_V2_50pct_gt_004m_any_gt_010m',
                         clearance_type='center_to_obstacle_m',collision_source='NOT_MEASURED',
                         contact_proxy=None,observations=156,action_mode=a.mode,
                         checkpoint=a.checkpoint,source_world=str(sc_path),
                         note='LiDAR+odom-only controller. Full true safety requires actual contacts/footprint monitoring.')
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')
            print('GAZEBO_EVAL_RESULT',json.dumps(summary),flush=True)
            self.stop_time=time.monotonic()

        def tick(self):
            if self.done:
                self.publish(0,0)
                if time.monotonic()-self.stop_time>0.5:rclpy.shutdown()
                return
            now=time.monotonic()
            if self.scan is None or self.odom is None:
                self.publish(0,0)
                if now-self.start_wall>15.0:self.get_logger().warn('Waiting for BOTH /scan and /odom; robot held stopped')
                return
            if s['dynamic'] and not self.actors_active:
                start=String();start.data=s['name'];self.actor_start_pub.publish(start)
                self.publish(0,0)
                if now-self.start_wall>15.0:
                    self.get_logger().warn('Dynamic actor not synchronized, robot held stopped')
                return
            if now-self.scan_received>1.0 or now-self.odom_received>1.0:
                self.publish(0,0)
                self.get_logger().warn('Stale /scan or /odom (>1s wall): fail-safe stop')
                return
            pose=odom_to_world(s['start'],self.initial_odom,self.odom)
            # S9_LAST2_PROXIMITY_STOP_V1: evaluation-only, independent of action policy.
            # Uses ground-truth solely to STOP and label trial when approaching a person.
            # The measured distance is center-to-center; it is NOT a physical contact sensor.
            if s.get('pedestrian_behavior') in ('slow_yield_20','independent_scripted_20'):  # S9_CROWD_V3_PROXIMITY_STOP
                person_names=[p['name'] for p in s['dynamic']]
                if 'burger' in self.gt_poses and all(n in self.gt_poses for n in person_names) and (now-self.gt_received)<1.0:
                    robot_gt=self.gt_poses['burger']
                    closest=min(math.hypot(self.gt_poses[n][0]-robot_gt[0],
                                           self.gt_poses[n][1]-robot_gt[1]) for n in person_names)
                    if closest < float(s['proximity_abort_center_m']):
                        self.get_logger().warn(f'PROXIMITY ABORT (not collision): center separation={closest:.3f}m')
                        self.finish('proximity_abort')
                        return
            # update previous action feature from odometry since last command.
            dgoal=float(math.hypot(s['goal'][0]-pose[0],s['goal'][1]-pose[1]))
            if self.prev_goal_distance is not None:
                self.progress=float(np.clip((self.prev_goal_distance-dgoal)/.22,-1.,1.))
            self.prev_goal_distance=dgoal
            if self.prev_pose is not None:self.distance+=float(np.linalg.norm(pose[:2]-self.prev_pose[:2]))
            self.prev_pose=pose.copy();self.recent.append(pose[:2].copy())
            moved=(sum(float(np.linalg.norm(self.recent[j]-self.recent[j-1]))
                       for j in range(1,len(self.recent))) if len(self.recent)>1 else 0.)
            self.stall=int(len(self.recent)==30 and moved<.08 and dgoal>.4)
            if dgoal<.20:
                self.finish('goal');return
            if self.steps>=a.max_steps:
                self.finish('timeout');return
            if self.stall and self.steps>50:
                self.finish('deadlock');return
            obs=make_obs(self.scan,pose,s['goal'],self.twist,self.prev_action,
                         self.prev_intervention,self.prev_slack,self.progress,self.stall,self.prev_scan)
            if obs.shape!=(156,) or not np.all(np.isfinite(obs)):
                self.finish('invalid_observation');return
            action=np.empty((0,),dtype=np.float32) if a.mode=='fixed' else self.agent.act(obs,deterministic=True)[0]
            output=controller_step(self.scan,pose,s['goal'],action,a.mode)
            self.prev_scan=self.scan.copy()
            self.publish(output['v'],output['w'])
            self.prev_action[:]=[output['a'],output['b']]
            self.prev_intervention[:]=[output['v']-output['v_nom'],output['w']-output['w_nom']]
            self.prev_slack=output['slack']
            # Ground truth is for EVALUATION ONLY. No GT pose fed to action selection.
            have_gt='burger' in self.gt_poses and (now-self.gt_received)<1.0
            truepos=self.gt_poses.get('burger') if have_gt else pose[:2]
            if s['dynamic']:
                dyn={d['name']:self.gt_poses[d['name']]
                     for d in s['dynamic'] if d['name'] in self.gt_poses} if have_gt else {}
                for name,pos in dyn.items():
                    self.actor_positions[name].append(pos)
                    self.actor_tracks.append(dict(step=self.steps+1,
                        sim_time_s=self.get_clock().now().nanoseconds*1e-9,
                        person=name,x_m=pos[0],y_m=pos[1],robot_x_m=truepos[0],
                        robot_y_m=truepos[1],
                        center_to_person_surface_m=math.hypot(pos[0]-truepos[0],pos[1]-truepos[1])-
                            next(d['radius'] for d in s['dynamic'] if d['name']==name)))
                cl=full_clearance(truepos,s,dyn) if have_gt else float('nan')
            else:
                cl=static_clearance(truepos,s)
            src='gazebo_model_states' if have_gt else 'odom_proxy'
            if math.isfinite(cl):
                self.min_clearance=min(self.min_clearance,cl)
                if cl<.25:self.violation_steps+=1
            self.steps+=1
            self.rows.append(dict(step=self.steps,sim_time_s=self.get_clock().now().nanoseconds*1e-9,
                x_odom_world=float(pose[0]),y_odom_world=float(pose[1]),yaw=float(pose[2]),
                goal_distance_m=dgoal,alpha=output['alpha'],b=output['b'],gate=output['gate'],
                v_cmd=output['v'],w_cmd=output['w'],v_nom=output['v_nom'],w_nom=output['w_nom'],
                v_odom=float(self.twist[0]),w_odom=float(self.twist[1]),slack=output['slack'],
                speed_limit=output['speed_limit'],goal_sector_range=output['goal_sector_range'],
                clearance_m=cl,clearance_source=src,violated_025=(int(cl<.25) if math.isfinite(cl) else ''),
                scan_min_m=float(np.min(self.scan))))

    rclpy.init()
    node=Bridge()
    try:rclpy.spin(node)
    except KeyboardInterrupt:node.get_logger().info('Interrupted; stopping robot')
    finally:
        if not node.done:node.finish('interrupted')
        for _ in range(5):node.publish(0,0);time.sleep(.05)
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
