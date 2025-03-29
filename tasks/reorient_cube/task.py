from __future__ import annotations

from leap_hand.leap import LEAPHandLeft
from tasks.reorient_cube.config import ReorientCubeEnvConfig
from utils import random_quaternion, unique_cube_rotations_3d, sample_rotations, batched_randint, batched_quat_diff

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common

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

class EnvDextreme(BaseEnv):
    def get_state_dict(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        pass
        
    def set_state_dict(self, env_state):
        pass
        
@register_env("ReorientCube-v0")
class ReorientCubeEnv(BaseEnv):
    """Environment for in-hand cube reorientation with LEAPHandLeft robot."""
    
    # Only one supported robot: LEAPHandLeft (16 DOFs)
    SUPPORTED_ROBOTS = ["leap_hand_left"]
    
    # Robot agent type
    agent: LEAPHandLeft
    
    # Scene parameters
    hand_elevation = 0.5 # The height of the hand above the tabletop (m?)
    
    # Simulation config
    sim_freq = 120 # This is supposed to be 60 in DeXtreme
    control_freq = 60 # This is supposed to be 30 in DeXtreme
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq = self.sim_freq,
            control_freq = self.control_freq,
        )
        
    def __init__(self, *args, robot_uids="leap_hand_left", obs_mode="state_dict", num_envs=1, config=ReorientCubeEnvConfig(), **kwargs):
        self.config = config
        
        # Create uninitialized goal state buffers
        self.goal_p = torch.empty((num_envs, 3), dtype=torch.float, device=self._sim_device)
        self.goal_q = torch.empty((num_envs, 4), dtype=torch.float, device=self._sim_device)
        
        # A list of all possible 90-degree cube rotations in 3D.
        self.rotations_pool = unique_cube_rotations_3d()
        
        # Allocate buffers
        # achieved_success - which subenvs succeeded in the last step (only success after hold, if applicable)
        # goal_achievement_timer - number of timesteps since the last goal was reached (progress_buf in DeXterme)
        # prev_targets - previous joint position targets
        # duration_goal_held - the number of steps the goal has been held (hold_count_buf in DeXtreme)
        # successes - the number of consecutive successes in each subenv since the last episode reset
        # last_actions - the last actions taken
        self.achieved_success = torch.ones(num_envs, dtype=torch.long, device=self._sim_device)
        self.prev_targets = torch.zeros((num_envs, config.dofs), dtype=torch.float, device=self._sim_device)
        self.goal_achievement_timer = torch.zeros(num_envs, dtype=torch.long, device=self._sim_device)
        self.goal_hold_timer = torch.zeros(num_envs, dtype=torch.long, device=self._sim_device)
        self.successes = torch.zeros(num_envs, dtype=torch.float, device=self._sim_device)
        self.last_actions = torch.zeros((num_envs, config.dofs), dtype=torch.float, device=self._sim_device)
        
        super().__init__(*args, robot_uids=robot_uids, obs_mode=obs_mode, num_envs=num_envs, **kwargs)
        
    def _load_agent(self, options: dict):
        """Override the initial pose of the robot hand."""
        initial_hand_pose = sapien.Pose(p=[0, 0, self.hand_elevation],
                                        q=[0, 1, 0, 0])
        
        super()._load_agent(options, initial_hand_pose)
        
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
        if self.config.sample_so3:
            self.goal_q[env_ids] = random_quaternion(env_ids, rng=self._batched_episode_rng)
        else:
            self.goal_q[env_ids] = sample_rotations(env_ids, self.rotations_pool, rng=self._batched_episode_rng)

        # Update the goals in the visualizer
        goal_pose = Pose.create_from_pq(p=[-0.1 + 0.045, 0.01 + 0.045, 0.505 + 0.045], q=self.goal_q)
        self.goal.set_pose(goal_pose)
        
    def _initialize_episode(self, env_ids: torch.Tensor, options: dict):
        """Set the initial states of all non-static objects, including the robot."""
        with torch.device(self.device):
            # Resample new goals for subenvs getting reset
            self.reset_goals(env_ids)
            
            # Reset buffers
            self.goal_hold_timer[env_ids] = 0
            self.last_actions[env_ids] = 0
            
            if self.config.use_adr and len(env_ids) == self.num_envs:
                # TODO: Why does DeXtreme do this?
                self.goal_achievement_timer = batched_randint(0, self.config.stuck_timeout * self.control_freq,
                                                              dtype=torch.long, device=self.device,
                                                              rng=self._batched_episode_rng)
            else:
                self.goal_achievement_timer[env_ids] = 0
    
    def _get_obs_extra(self, info):
        # TODO: There are some more observations we need
        obs = {
            #"object_position_with_noise": None,
            #"object_orientation_with_noise": None,
            "target_position": self.goal.pose.p,
            "target_orientation": self.goal.pose.q,
            "relative_target_orientation": batched_quat_diff(self.goal.pose.q, self.cube.pose.q),
            "last_actions": self.last_actions.clone(),
            "hand_joint_angles": self.agent.controller.qpos,
            "hand_joint_velocities": self.agent.robot.qvel,
            #"stochastic_delays": None,
            #"fingertip_torques": None,
            #"hand_joints_generalized_forces": None,
            #"object_scale": None,
            "object_mass": self.cube.mass,
            #"object_friction": None,
            "object_linear_velocity": self.cube.linear_velocity,
            "object_angular_velocity": self.cube.angular_velocity,
            "object_position": self.cube.pose.p,
            "object_orientation": self.cube.pose.q,
            #"random_forces": None,
            #"domain_randomization_params": None,
            #"gravity_vector": None,
            #"rotation_distances": None,
            #"hand_scale": None,
        }
        
        fingertip_positions = []
        fingertip_rotations = []
        fingertip_linear_velocities = []
        fingertip_angular_velocities = []
        fingertip_forces = []
        
        for fingertip_link_name in ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]:
            fingertip_link = self.agent.robot.find_link_by_name(fingertip_link_name)
            
            fingertip_positions.append(fingertip_link.pose.p)
            fingertip_rotations.append(fingertip_link.pose.q)
            fingertip_linear_velocities.append(fingertip_link.linear_velocity)
            fingertip_angular_velocities.append(fingertip_link.angular_velocity)
            fingertip_forces.append(fingertip_link.get_net_contact_forces())
            
        obs["fingertip_positions"] = torch.cat(fingertip_positions)
        obs["fingertip_rotations"] = torch.cat(fingertip_rotations)
        obs["fingertip_linear_velocities"] = torch.cat(fingertip_linear_velocities)
        obs["fingertip_angular_velocities"] = torch.cat(fingertip_angular_velocities)
        obs["fingertip_forces"] = torch.cat(fingertip_forces)
        
        return obs
        
    def _before_control_step(self):
        # Resample new goals in subenvs that just reached success in the last step
        reset_goal_env_ids = self.achieved_success.nonzero().squeeze(-1)
        self.reset_goals(reset_goal_env_ids)
        
        # Apply random forces
        self.apply_random_forces()

    ### TODO
    def apply_random_forces(self):
        pass
      
    def _after_control_step(self):
        self.goal_achievement_timer += 1
        
    def evaluate(self):
        # Compute error terms
        position_error = torch.norm(self.cube.pose.p - self.goal.pose.p, dim = -1)
        orientation_error = common.quat_diff_rad(self.cube.pose.q, self.goal_q)
        
        # Check for success
        # TODO: Do we need torch.abs here or is it already non-negative?
        reached_goal = torch.abs(orientation_error) <= self.config.success_tolerance
        
        self.goal_hold_timer = torch.where(reached_goal, self.goal_hold_timer + 1, 0)
        self.achieved_success = self.goal_hold_timer > self.config.success_hold_duration
        self.successes += self.achieved_success
        
        # Check for dropped cube
        dropped_cube = position_error >= self.config.drop_distance
        
        # Get whether environments are in a success state (achieved maximum consecutive goals)
        if self.config.max_consecutive_successes > 0:
            self.goal_achievement_timer = torch.where(reached_goal, 0, self.goal_achievement_timer)
            succeeded = self.successes >= self.config.max_consecutive_successes
        else:
            succeeded = None
        
        # Check for goal achievment timeout
        timed_out = self.goal_achievement_timer >= self.config.stuck_timeout * self.control_freq - 1
        
        # Get whether environments are in a failure state
        failed = torch.bitwise_or(dropped_cube, timed_out)
        
        # Return info
        info = {
            "fail": failed.to(torch.int),
            "goal": self.achieved_success,
            "drop": dropped_cube,
            "timeout": timed_out,
            "position_error": position_error,
            "orientation_error": orientation_error,
        }
        
        if succeeded is not None:
            info["success"] = succeeded.to(torch.int)
            
        return info

    def compute_normalized_dense_reward(self, obs, action, info):
        # Reward for reorienting the cube toward the goal orientation
        orientation_reward = self.config.reward.orientation_gain / (info["orientation_error"] + self.config.reward.orientation_eps)
        
        # Reward for keeping the cube close to a fixed goal position
        position_reward = self.config.reward.position_gain * info["position_error"]
        
        # Penalty for large actions
        # These are the raw actions passed to env.step, not preprocessed by the controller.
        action_penalty = self.config.reward.action_penalty * torch.sum(action ** 2, dim = -1)
        
        # Penalty for rapid changes in joint targets
        # For the built-in PID joint position controller, you can get the joint targets as follows:
        curr_targets = self.agent.controller.get_state()["target_qpos"]
        action_delta_penalty = self.config.reward.action_delta_penalty * torch.sum((curr_targets - self.prev_targets) ** 2, dim = -1)
        
        # Penalty for large joint velocities
        velocity_penalty = self.config.reward.velocity_penalty * torch.sum(obs["agent"]["qvel"] ** 2, dim = -1)
        
        # Bonus for reaching a goal
        success_reward = self.config.reward.success_bonus * info["goal"]
        
        # Penalty for dropping the cube
        drop_penalty = self.config.reward.drop_penalty * info["drop"]
        
        # Penalty for not reaching the goal in time
        timeout_penalty = self.config.reward.timeout_penalty * info["timeout"]
        
        # Update the previous targets 
        self.prev_targets = curr_targets
        self.last_actions = action
        
        return (
            orientation_reward +\
            position_reward +\
            action_penalty +\
            action_delta_penalty +\
            velocity_penalty +\
            success_reward +\
            drop_penalty +\
            timeout_penalty
        )