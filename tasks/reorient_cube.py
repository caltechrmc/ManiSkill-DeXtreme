from __future__ import annotations

from leap_hand.leap import LEAPHandLeft
from utils import random_quaternion, unique_cube_rotations_3d, sample_rotations

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.pose import Pose

import sapien
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

def build_cube(scene: ManiSkillScene):
    """Build the competition cube."""
    builder = scene.create_actor_builder()

    # Create the collision body
    builder.add_nonconvex_collision_from_file(
        "models/cube_l_rescaled_recentered.stl",
    )
    
    # Create the visual body
    builder.add_visual_from_file(
        "models/cube_l_rescaled_recentered.stl",
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
    )

    builder.initial_pose = sapien.Pose(p=[-0.1 + 0.045, 0.01 + 0.045, 0.505 + 0.045],
                                       q=[1, 0, 0, 0])
    
    return builder.build(name = "cube")

def build_goal(scene):
    """Build the target cube."""
    builder = scene.create_actor_builder()
    
    # Create the visual body
    builder.add_visual_from_file(
        "models/cube_l_rescaled_recentered.stl",
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
    )

    builder.initial_pose = sapien.Pose(p=[-0.1 + 0.045, 0.01 + 0.045, 0.505 + 0.045],
                                       q=[1, 0, 0, 0])
    
    return builder.build_static(name = "goal")

@register_env("ReorientCube-v0")
class ReorientCubeEnv(BaseEnv):
    """Environment for in-hand cube reorientation with LEAPHandLeft robot."""
    
    # Only one supported robot: LEAPHandLeft (16 DOFs)
    SUPPORTED_ROBOTS = ["leap_hand_left"]
    
    # Robot agent type
    agent: LEAPHandLeft
    
    # Model constants
    dofs = 16
    
    # Scene parameters
    hand_elevation = 0.5 # The height of the hand above the tabletop (m?)
    
    # If true, goals are selected from SO(3). If False, they are selected
    # uniformly from all possible 90° cube rotations.
    sample_so3 = False
    
    # Simulation config
    sim_freq = 120
    control_freq = 60
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq = self.sim_freq,
            control_freq = self.control_freq,
        )
        
    def __init__(self, *args, robot_uids="leap_hand_left", obs_mode="state_dict", num_envs=1, **kwargs):
        # Create uninitialized goal state buffers
        self.goal_p = torch.empty((num_envs, 3), dtype=torch.float, device=self._sim_device)
        self.goal_q = torch.empty((num_envs, 4), dtype=torch.float, device=self._sim_device)
        
        # A list of all possible 90-degree cube rotations in 3D.
        self.rotations_pool = unique_cube_rotations_3d()
        
        super().__init__(*args, robot_uids=robot_uids, obs_mode=obs_mode, num_envs=num_envs, **kwargs)
        
        #self.time_since_last_success = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        #self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
    def _load_agent(self, options: dict):
        """Override the initial pose of the robot hand."""
        hand_pose = sapien.Pose(p=[0, 0, self.hand_elevation],
                                q=[0, 1, 0, 0])
        
        super()._load_agent(options, hand_pose)
        
    def _load_scene(self, options: dict):
        """Build all of the non-agent objects."""
        
        # Create the ground
        self.ground = build_ground(self.scene)
        
        # Create the cube
        self.cube = build_cube(self.scene)
        
        # Create the goal
        self.goal = build_goal(self.scene)
    
    def reset_goals(self, env_ids):
        """Resample new goals for the specified environment IDs."""
        
        # Resample the goal poses
        if self.sample_so3:
            self.goal_q[env_ids] = random_quaternion(env_ids, rng=self._batched_episode_rng)
        else:
            self.goal_q[env_ids] = sample_rotations(env_ids, self.rotations_pool, rng=self._batched_episode_rng)

        # Update the goals in the visualizer
        goal_pose = Pose.create_from_pq(p=[-0.1 + 0.045, 0.01 + 0.045, 0.505 + 0.045], q=self.goal_q)
        self.goal.set_pose(goal_pose)
        
    def _initialize_episode(self, env_ids: torch.Tensor, options: dict):
        """Set the initial states of all non-static objects, including the robot."""
        with torch.device(self.device):
            # Resample new goals where needed
            self.reset_goals(env_ids)
            
            # Reset buffers
            #self.consecutive_successes[env_ids] = 0
            #self.time_since_last_success[env_ids] = 0