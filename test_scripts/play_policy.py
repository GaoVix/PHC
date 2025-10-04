# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import argparse
# from isaaclab.app import AppLauncher
import sys
from isaacgym import gymapi

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--motion_file", type=str, default="sample_data/amass_isaac_standing_upright_slim.pkl", help="Path to motion file to load.")
parser.add_argument("--policy_path", type=str, default="output/HumanoidIm/phc_3/Humanoid.pth", help="Path to policy checkpoint file.")
parser.add_argument("--action_offset_file", type=str, default="phc/data/action_offset_smpl.pkl", help="Path to action offset file.")
parser.add_argument("--humanoid_type", type=str, default="smpl", choices=["smpl", "smplx"], help="Type of humanoid model to use.")
parser.add_argument("--num_motions", type=int, default=10, help="Number of motions to load from motion library.")
# append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

"""Rest everything follows."""

import torch


# import carb
# import imageio
# from carb.input import KeyboardEventType



##
# Pre-defined configs
##
# from isaaclab_assets import CARTPOLE_CFG  # isort:skip
# from phc.assets.smpl_config import SMPL_Upright_CFG, SMPL_CFG, SMPLX_Upright_CFG, SMPLX_CFG

# from smpl_sim.utils.rotation_conversions import xyzw_to_wxyz, wxyz_to_xyzw
# from collections.abc import Sequence
# from phc.utils.flags import flags
# from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
# from phc.utils.motion_lib_base import FixHeightMode
# from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
# from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
# from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp
# from phc.utils.isaacgym_humanoid_funcs import compute_humanoid_observations_smpl_max, compute_imitation_observations_v6 
from rl_games.algos_torch import torch_ext
# from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
# from rl_games.algos_torch.players import rescale_actions
# import joblib
# from easydict import EasyDict
# import numpy as np
# import copy
# from scipy.spatial.transform import Rotation as sRot
# import time

class Env:
    def __init__(self, cfg):
        self.headless = cfg["headless"]
        if self.headless == False and not flags.no_virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=(1800, 990), visible=True)
            self.virtual_display.start()

        self.gym = gymapi.acquire_gym()
        self.paused = False
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.state_record = defaultdict(list)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1
        # if flags.server_mode:
        # self.graphics_device_id = self.device_id

        self.num_envs = cfg["env"]["num_envs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["control"].get("decimation", 2)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        self.create_viewer()

    def pre_physics_step(self, actions):

        self.actions = actions.to(self.device).clone()

        if self.collect_dataset:
            self.clean_actions = actions.to(self.device).clone()

            if self.add_action_noise:
                noise = torch.normal(mean=0.0, std=float(self.action_noise_std), size = actions.shape, device=self.device)
                self.actions += noise

        if len(self.actions.shape) == 1:
            self.actions = self.actions[None, ]

        clip_actions = 10
        self.actions = torch.clip(self.actions, -clip_actions, clip_actions).to(self.device) # -10 , 10
                
        return

    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker.urdf", asset_options)
        
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

        return

    def create_envs(self, ):
        self._marker_handles = [[] for _ in range(num_envs)]
        self._load_marker_asset()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        
        asset_root = self.cfg.robot.asset["assetRoot"]
        asset_file = self.cfg.robot.asset["assetFileName"]
        self.humanoid_masses = []

        self.humanoid_limb_and_weights = []
        xml_asset_path = os.path.join(asset_root, asset_file)
        
        robot_file = os.path.join(asset_root, self.cfg.robot.asset.urdfFileName)
        asset_root = os.path.dirname(robot_file) # use urdf file. 
        asset_file = os.path.basename(robot_file)
        sk_tree = SkeletonTree.from_mjcf(xml_asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        #asset_options.fix_base_link = True
        asset_options.replace_cylinder_with_capsule = True
        
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        
        # motor_efforts = [prop.motor_effort for prop in actuator_props]
        motor_efforts = [360] * 19

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self.cfg.robot.right_foot_name)
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self.cfg.robot.left_foot_name)
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)
        self.humanoid_shapes = torch.tensor(np.zeros((num_envs, 10))).float().to(self.device)
        self.humanoid_assets = [humanoid_asset] * num_envs
        self.skeleton_trees = [sk_tree] * num_envs

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)
        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_asset_joints = self.gym.get_asset_joint_count(humanoid_asset)
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        max_agg_bodies, max_agg_shapes = 160, 160
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, self.humanoid_assets[i])
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)
        print("Humanoid Weights", self.humanoid_masses[:10])

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])

        for j in range(self.num_dof):
            
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            elif dof_prop['lower'][j] == dof_prop['upper'][j]:
                print("Warning: DOF limits are the same")
                if dof_prop['lower'][j] == 0:
                    self.dof_limits_lower.append(-np.pi)
                    self.dof_limits_upper.append(np.pi)
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
        
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop['effort'], device = self.device)
        
        self._process_dof_props(dof_prop)

        self._build_pd_action_offset_scale()
        return
    
    
    
def main():

    config = {
        "policy_path": "/mnt/Exp_HDD/projects/phc/output/HumanoidIm/unitree_g1_pnn2/Humanoid.pth",
        "xml_path": "./robots/go2/scene.xml",
        "simulation_duration": 3000,
        "simulation_dt": 0.002,
        "control_decimation": 10,
        "kps": 50.0,
        "kds": 0.1,
        "default_angles": [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
        "ang_vel_scale": 1.0,
        "dof_pos_scale": 1.0,
        "dof_vel_scale": 1.0,
        "action_scale": 0.2,
        "cmd_scale": 1.0,
        "num_actions": 12,
        "num_obs": 48,
        "cmd_init": [1.5, 0.0, 0.0]
    }
    
    
    check_points = torch_ext.load_checkpoint(config['policy_path'])
    pnn = load_pnn(check_points, num_prim = 3, has_lateral = False, activation = "silu", device = 'cuda')
    running_mean, running_var = check_points['running_mean_std']['running_mean'], check_points['running_mean_std']['running_var']
    
    # action_offset = joblib.load(args_cli.action_offset_file)
        
    # pd_action_offset = action_offset[0]
    # pd_action_scale = action_offset[1]
    
    # time = 0 
    # obs_dict, extras = env.reset()
    # while True:
    #     self_obs, task_obs = obs_dict["self_obs"], obs_dict["task_obs"]
    #     full_obs = torch.cat([self_obs, task_obs], dim = -1)
    #     full_obs = ((full_obs - running_mean.float()) / torch.sqrt(running_var.float() + 1e-05))
    #     full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
        
        
    #     with torch.no_grad():
    #         actions, _ = pnn(full_obs, idx=0)
    #         actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
    #         actions = actions * pd_action_scale + pd_action_offset
    #         actions = actions[:, env.gym_to_sim_dof]
        
    #     obs_dict, _, _, _, _ = env.step(actions)
    


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # simulation_app.close()
    
    