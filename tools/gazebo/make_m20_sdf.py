#!/usr/bin/env python3
"""Generate the Gazebo (gz-sim Harmonic) M20 model from the DeepRobotics URDF: tools/gazebo/models/m20/model.sdf.

Milestone-1 drive model: the body is ONE rigid link (box collision + the real meshes as visuals, legs and wheels
posed at the standing stance, no articulation); a VelocityControl plugin drives it kinematically from a Twist, like
the Godot sim's M20. Sensors match tools/godot's vehicle.json so the ROS stack sees the same geometry:
  two hemispherical lidars (RoboSense Airy stand-in: 360 x 0..90 deg, looking UP from the mounts, 10 Hz)
  one RGB-D camera (640x480, 90 deg hfov, 10 Hz) on the front mount, an IMU on base_link,
  OdometryPublisher = ground truth (odom -> base_link, the glue node derives drift and /odom_true from it).
    python3 tools/gazebo/make_m20_sdf.py            (numpy only; run from the repo root)
"""
import json
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "m20")
URDF = os.path.join(MODEL_DIR, "M20_source.urdf")

# the standing stance (tools/godot vehicle.json "controller.stance"), radians
STANCE = {"fl_hipx_joint": 0.0, "fl_hipy_joint": -0.7, "fl_knee_joint": 1.4,
          "fr_hipx_joint": 0.0, "fr_hipy_joint": -0.7, "fr_knee_joint": 1.4,
          "hl_hipx_joint": 0.0, "hl_hipy_joint": 0.7, "hl_knee_joint": -1.4,
          "hr_hipx_joint": 0.0, "hr_hipy_joint": 0.7, "hr_knee_joint": -1.4}
MOUNTS = {"front_camera_link": (0.37646, 0.0, 0.03738), "front_lidar_link": (0.32028, 0.0, -0.013),
          "rear_lidar_link": (-0.32028, 0.0, -0.013)}
BODY_BOX = (0.86, 0.56, 0.30)          # footprint the stack is tuned for (Nav2 footprint 0.86 x 0.56)
BASE_Z = 0.43                           # base_link height above the floor when standing (bringup: floor at z -0.43)
LIDAR_RINGS, LIDAR_COLS = 32, 450       # sim resolution (the real Airy: 96 x 900; the stack only uses a z-band)
CAM_W, CAM_H, CAM_HFOV = 640, 480, math.radians(90.0)
# The sensors sit INSIDE the body (lidars at |x| 0.32 of a 0.86 m body): without this the lidars return ~5k points
# on the robot itself (measured: r < 0.3 m), the costmap marks the footprint lethal and Nav2 can't plan or move.
# Visibility: the robot's visuals carry flag bit 1; the sensors' masks keep bit 0 only -> they never see the robot.
ROBOT_VIS_FLAGS, SENSOR_VIS_MASK = "0x00000002", "0x00000001"


def rot(axis, q):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    x, y, z = a; c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rpy_from(R):
    sy = -R[2, 0]
    p = math.asin(max(-1.0, min(1.0, sy)))
    r = math.atan2(R[2, 1], R[2, 2]); y = math.atan2(R[1, 0], R[0, 0])
    return r, p, y


def parse_urdf():
    root = ET.parse(URDF).getroot()
    links = {}
    for l in root.findall("link"):
        v = l.find("visual")
        mesh = v.find("geometry/mesh").get("filename") if v is not None and v.find("geometry/mesh") is not None else None
        o = v.find("origin") if v is not None else None
        links[l.get("name")] = {"mesh": os.path.basename(mesh) if mesh else None,
                                "vxyz": [float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()],
                                "vrpy": [float(x) for x in (o.get("rpy") if o is not None else "0 0 0").split()]}
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin"); ax = j.find("axis")
        joints[j.get("name")] = {"parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                                 "xyz": [float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()],
                                 "rpy": [float(x) for x in (o.get("rpy") if o is not None else "0 0 0").split()],
                                 "axis": [float(x) for x in (ax.get("xyz") if ax is not None else "1 0 0").split()],
                                 "type": j.get("type")}
    return links, joints


def rpy_mat(r, p, y):
    return rot((0, 0, 1), y) @ rot((0, 1, 0), p) @ rot((1, 0, 0), r)


def link_poses(links, joints):
    """World-of-base pose of every link at the stance (visual-only chain)."""
    poses = {"base_link": (np.zeros(3), np.eye(3))}
    pending = dict(joints)
    while pending:
        for n, j in list(pending.items()):
            if j["parent"] in poses:
                p, R = poses[j["parent"]]
                Rj = rpy_mat(*j["rpy"]); pj = p + R @ np.asarray(j["xyz"])
                Rc = R @ Rj
                q = STANCE.get(n, 0.0)
                if j["type"] in ("revolute", "continuous") and q:
                    Rc = Rc @ rot(j["axis"], q)
                poses[j["child"]] = (pj, Rc); del pending[n]
    return poses


def pose_str(p, R):
    r, pi, y = rpy_from(R)
    return f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {r:.5f} {pi:.5f} {y:.5f}"


def main():
    links, joints = parse_urdf()
    poses = link_poses(links, joints)
    visuals = []
    for name, (p, R) in poses.items():
        L = links.get(name, {})
        if not L.get("mesh"):
            continue
        pv = p + R @ np.asarray(L["vxyz"]); Rv = R @ rpy_mat(*L["vrpy"])
        visuals.append(f"""      <visual name="{name}_visual">
        <pose>{pose_str(pv, Rv)}</pose>
        <geometry><mesh><uri>model://m20/meshes/{L['mesh']}</uri></mesh></geometry>
        <material><ambient>0.25 0.25 0.27 1</ambient><diffuse>0.35 0.35 0.38 1</diffuse></material>
        <visibility_flags>{ROBOT_VIS_FLAGS}</visibility_flags>
      </visual>""")
    bx, by, bz = BODY_BOX
    fl, rl, cam = MOUNTS["front_lidar_link"], MOUNTS["rear_lidar_link"], MOUNTS["front_camera_link"]

    def lidar(name, xyz):
        return f"""      <sensor name="{name}_lidar" type="gpu_lidar">
        <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
        <topic>/m20/{name}_lidar</topic>
        <gz_frame_id>{name}_lidar_link</gz_frame_id>
        <update_rate>10</update_rate>
        <lidar>
          <scan>
            <horizontal><samples>{LIDAR_COLS}</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
            <vertical><samples>{LIDAR_RINGS}</samples><resolution>1</resolution><min_angle>0.0</min_angle><max_angle>1.5533</max_angle></vertical>
          </scan>
          <range><min>0.1</min><max>60.0</max><resolution>0.01</resolution></range>
          <noise><type>gaussian</type><mean>0.0</mean><stddev>0.01</stddev></noise>
          <visibility_mask>{SENSOR_VIS_MASK}</visibility_mask>
        </lidar>
        <always_on>1</always_on><visualize>0</visualize>
      </sensor>"""

    sdf = f"""<?xml version="1.0"?>
<!-- GENERATED by tools/gazebo/make_m20_sdf.py from M20_source.urdf (DeepRobotics URDF_model, BSD-3). Do not edit. -->
<sdf version="1.9">
  <model name="m20">
    <static>false</static>
    <link name="base_link">
      <pose>0 0 {BASE_Z} 0 0 0</pose>
      <inertial>
        <mass>34.5</mass>
        <inertia><ixx>0.56</ixx><iyy>0.53</iyy><izz>0.088</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="body_collision">
        <pose>0 0 {-BASE_Z / 2 + 0.02:.3f} 0 0 0</pose>
        <geometry><box><size>{bx} {by} {BASE_Z + 0.04:.3f}</size></box></geometry>
        <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface>
      </collision>
{chr(10).join(visuals)}
{lidar("front", fl)}
{lidar("rear", rl)}
      <sensor name="front_camera" type="rgbd_camera">
        <pose>{cam[0]} {cam[1]} {cam[2]} 0 0 0</pose>
        <topic>/m20/camera</topic>
        <gz_frame_id>front_camera_link</gz_frame_id>
        <update_rate>10</update_rate>
        <camera>
          <horizontal_fov>{CAM_HFOV:.5f}</horizontal_fov>
          <image><width>{CAM_W}</width><height>{CAM_H}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>30.0</far></clip>
          <depth_camera><clip><near>0.1</near><far>10.0</far></clip></depth_camera>
          <visibility_mask>{SENSOR_VIS_MASK}</visibility_mask>
        </camera>
        <always_on>1</always_on><visualize>0</visualize>
      </sensor>
      <sensor name="imu" type="imu">
        <topic>/m20/imu</topic>
        <gz_frame_id>base_link</gz_frame_id>
        <update_rate>100</update_rate>
        <always_on>1</always_on>
      </sensor>
    </link>
    <!-- kinematic planar drive from a Twist (body frame), like the Godot M20 -->
    <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
      <topic>/m20/cmd_vel</topic>
    </plugin>
    <!-- the model's TRUE world pose (/model/m20/pose, Pose_V): the glue node makes /odom_true and odom from it -->
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_model_pose>true</publish_model_pose>
      <publish_link_pose>false</publish_link_pose>
      <publish_collision_pose>false</publish_collision_pose>
      <publish_visual_pose>false</publish_visual_pose>
      <publish_sensor_pose>false</publish_sensor_pose>
      <publish_nested_model_pose>false</publish_nested_model_pose>
      <use_pose_vector_msg>true</use_pose_vector_msg>
      <update_frequency>50</update_frequency>
    </plugin>
    <!-- body-frame velocity (odom = the spawn pose); the glue node uses its twist -->
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_topic>/m20/odometry</odom_topic>
      <tf_topic>/m20/odom_tf</tf_topic>   <!-- keep its odom-relative Pose_V OFF /model/m20/pose (the true pose) -->
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <odom_publish_frequency>50</odom_publish_frequency>
      <dimensions>3</dimensions>
    </plugin>
  </model>
</sdf>
"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "model.sdf"), "w", newline="\n") as f:
        f.write(sdf)
    with open(os.path.join(MODEL_DIR, "model.config"), "w", newline="\n") as f:
        f.write('<?xml version="1.0"?>\n<model>\n  <name>m20</name>\n  <version>1.0</version>\n  <sdf version="1.9">model.sdf</sdf>\n'
                '  <author><name>DeepRobotics (URDF); ROS2_Nav2_Mako (SDF)</name></author>\n'
                '  <description>Deep Robotics M20, posed body with sensors, kinematic drive</description>\n</model>\n')
    print("wrote", os.path.join(MODEL_DIR, "model.sdf"), "visuals:", len(visuals))


if __name__ == "__main__":
    main()
