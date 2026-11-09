#!/usr/bin/env python3
"""Moves kinematic circular actors by SetEntityState; evaluation only.

This is scripted-trajectory / kinematic obstacle motion, not reactive physics.
"""
import argparse,json,math,time
from pathlib import Path
from transfer_core import load_scenario

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',required=True);p.add_argument('--rate',type=float,default=10.)
    args=p.parse_args()
    path=Path(args.scenario)
    if not path.is_file():path=Path(__file__).resolve().parent/'scenarios'/f'{args.scenario}.json'
    s=load_scenario(path)
    if not s['dynamic']:
        print('No moving actors for this scenario.');return
    import rclpy
    from rclpy.node import Node
    from gazebo_msgs.srv import SetEntityState
    from std_msgs.msg import Bool, String
    class Actor(Node):
        def __init__(self):
            super().__init__('cbf_v31_dynamic_actors')
            if self.has_parameter('use_sim_time'):
                from rclpy.parameter import Parameter
                self.set_parameters([Parameter('use_sim_time', value=True)])
            else:
                self.declare_parameter('use_sim_time', True)
            self.cli=None;self.pending={};self.clock0=None;self.last_warn=0;self.started=False
            self.active_pub=self.create_publisher(Bool,'/cbf_v31/actors_active',10)
            self.create_subscription(String,'/cbf_v31/episode_start',self.on_start,10)
            for service in ['/set_entity_state','/gazebo/set_entity_state']:
                c=self.create_client(SetEntityState,service)
                if c.wait_for_service(timeout_sec=1.0):
                    self.cli=c;self.get_logger().info('Using '+service);break
            if self.cli is None:raise RuntimeError('SetEntityState service missing; add gazebo_ros_state world plugin')
            self.create_timer(1.0/args.rate,self.tick)
        def on_start(self,msg):
            if msg.data==s['name'] and not self.started:
                self.started=True
                self.clock0=float(self.get_clock().now().nanoseconds)*1e-9
                self.get_logger().info('Synchronized dynamic episode START '+s['name'])
        def tick(self):
            now=float(self.get_clock().now().nanoseconds)*1e-9
            if now==0:return
            hb=Bool();hb.data=self.started;self.active_pub.publish(hb)
            if not self.started:return
            t=now-self.clock0
            for d in s['dynamic']:
                n=d['name'];old=self.pending.get(n)
                if old is not None and not old.done():
                    continue
                if old is not None:
                    try:
                        resp=old.result()
                        if not resp.success and time.monotonic()-self.last_warn>2:
                            self.get_logger().warn(f'Failed to move {n}: {resp.status_message}');self.last_warn=time.monotonic()
                    except Exception as exc:self.get_logger().error(str(exc))
                phase=2*math.pi*t/d['period_s']+d.get('phase',0.)
                x=d['x']+(d['amplitude']*math.sin(phase) if d['axis']=='x' else 0.)
                y=d['y']+(d['amplitude']*math.sin(phase) if d['axis']=='y' else 0.)
                req=SetEntityState.Request()
                req.state.name=n;req.state.pose.position.x=float(x);req.state.pose.position.y=float(y)
                req.state.pose.position.z=.24;req.state.pose.orientation.w=1.
                req.state.reference_frame='world'
                self.pending[n]=self.cli.call_async(req)
    rclpy.init();node=None
    try:
        node=Actor();rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
