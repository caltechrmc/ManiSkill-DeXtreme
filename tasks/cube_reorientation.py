from leap_hand.leap import LEAPHandLeft, LEAPHandRight
from rgmc.objects import RGMCCubeSize, get_cube_size
from utils import random_quaternion

import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from mani_skill.utils.building import actors
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

import numpy as np
import torch

from typing import Union

# The height of the hand above the tabletop (m?)
HAND_ELEVATION = 0.3

# The threshold to consider a target orientation reached (rad)
ORIENTATION_THRESHOLD = 0.1

# The maximum amount of time for which the object can be stuck
# in the same configuration, before task failure occurs (seconds).
TIME_LIMIT = 80

# The plane below which, if the cube falls, it is considered dropped,
# upon which task failure occurs.
DROP_HEIGHT = 1.1 * get_cube_size(RGMCCubeSize.SMALL) / 2

# Reward term coefficients
ORIENTATION_GAIN = 1.0 # Shaped reward to bring cube close to goal
POSITION_GAIN = -10.0 # Encourage the cube to stay in the hand
ACTION_PENALTY = -0.001 # Prevent actions that are too large
ACTION_DELTA_PENALTY = -0.25 # Prevent rapid changes in joint target
JOINT_VELOCITY_PENALTY = -0.003 # Stop fingers from moving too quickly
REACH_GOAL_BONUS = 250.0 # Large reward for getting the cube to the target

def build_cube(name: str, initial_pose: Pose, scene):
    builder = scene.create_actor_builder()

    # Create the collision body
    builder.add_nonconvex_collision_from_file(
        "rgmc_in_hand_manipulation_2025/models/cube_s.STL",
        scale = [0.001] * 3,
    )
    
    # Create the visual body
    builder.add_visual_from_file(
        "rgmc_in_hand_manipulation_2025/models/cube_s.STL",
        scale = [0.001] * 3,
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
    )

    builder.initial_pose = initial_pose
    
    return builder.build(name = name)
    
@register_env("ReorientCube-v1")
class CubeReorientationEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["leap_hand_left", "leap_hand_right"]
    
    agent: Union[LEAPHandLeft, LEAPHandRight]

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)
        ]
    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq = 60,
            control_freq = 30,
        )
    def __init__(self, *args, robot_uids="leap_hand_left", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    def _load_agent(self, options: dict):
        """Override the initial pose of the robot."""
        super()._load_agent(options, sapien.Pose(p=[0, 0, HAND_ELEVATION], q=[0, 1, 0, 0]))
        
    def _load_scene(self, options: dict):
        """
        Build all of the non-agent objects.
        
        Args:
            options (dict): The options dictionary passed to env.reset
        """
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        # Create the cube
        initial_pose = sapien.Pose(p=[0, 0, HAND_ELEVATION + 0.05], q=[1, 0, 0, 0])
        self.cube = build_cube("cube", initial_pose=initial_pose, scene=self.scene)

        self.time_since_reconfig = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        self.goal_p = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.goal_q = torch.empty((self.num_envs, 4), dtype=torch.float32, device=self.device)
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """
        Set the initial states of all non-static objects, including the robot.
        Called whenever env.reset is called.
        
        Options will include ADR_modes and ADR_vals. See Algorithm 2 on page 9 of Dextreme.
        
        Args:
            env_idx (torch.Tensor): A list of environment IDs that need initialization.
        """
        with torch.device(self.device):
            # The number of environments in needs of resetting. We are doing
            # a partial reset.
            num_envs = len(env_idx)
            
            # use the TableSceneBuilder to init all objects in that scene builder
            self.table_scene.initialize(env_idx)
            
            # Reset the time since last reconfiguration
            self.time_since_reconfig[env_idx] = 0
            
            # Sample new goal orientations from SO(3)
            self.goal_q[env_idx] = random_quaternion(num_envs)
            
    def evaluate(self):
        """
        Returns:
            info (dict): The info object returned by env.step().
        """
        # Check for success conditions
        orientation_error = common.quat_diff_rad(self.cube.pose.q, self.goal_q)
        
        task_success = orientation_error < ORIENTATION_THRESHOLD
        
        # Check for failure conditions
        dropped_cube = self.cube.pose.p[..., 2] < DROP_HEIGHT
        time_expired = self.time_since_reconfig > TIME_LIMIT
        
        task_failure = torch.bitwise_or(dropped_cube, time_expired)
        
        return {
            "success": task_success,
            "failure": task_failure,
        }
    
    def compute_normalized_dense_reward(self, obs, action, info):
        # Rotation Close to Goal
        orientation_error = common.quat_diff_rad(self.cube.pose.q, self.goal_q)
        orientation_reward = ORIENTATION_GAIN / (orientation_error + 0.1)
        
        # Position Close to Fixed Target
        position_error = torch.norm(self.cube.pose.p - self.goal_p, dim = -1)
        position_reward = POSITION_GAIN * position_error
        
        # Action Penalty
        action_reward = ACTION_PENALTY * torch.norm(action, dim = -1)
        
        # Action Delta Penalty
        # TODO: Implement this
        
        # Joint Velocity Penalty
        # TODO: Implement this
        
        # Reach Goal Bonus
        reach_goal_reward = REACH_GOAL_BONUS * (orientation_error < ORIENTATION_THRESHOLD)
        
        return orientation_reward + position_reward + action_reward + reach_goal_reward