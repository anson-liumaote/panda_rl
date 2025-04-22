# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import numpy as np
from isaaclab.utils.math import quat_rotate_inverse
import matplotlib.pyplot as plt

##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)

def base_height_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float = 0.2) -> torch.Tensor:
    """Penalize deviation from target base height"""
    # Extract the asset to access its properties
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Penalty on deviation from target height
    # asset.data.root_pos_w[:, 2] gives the z-coordinate of the root position in world frame
    height_penalty = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    
    return height_penalty

# class AnimationPositionReward(ManagerTermBase):
#     """Gait enforcing reward term for quadrupeds.
#     A class to read joint angles data from a text file and convert to PyTorch tensors.
#     The data format is expected to be space-separated values with 12 joint angles per line.
#     When reaching the end of file, it automatically loops back to the beginning.
#     Each pose is extended to 4096 identical instances.
#     """
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         """Initialize the term.
#         Args:
#             cfg: The configuration of the reward.
#             env: The RL environment instance.
#             file_path (str): Path to the joint angles text file
#         """
#         super().__init__(cfg, env)
#         self.std: float = cfg.params["std"]
#         self.velocity_threshold: float = cfg.params["velocity_threshold"]
#         self.file_path: str = cfg.params["file_path"]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
#         self.current_line = 0
        
#         # Store the device from the asset for later use
#         # self.device = self.asset.data.joint_pos.device
        
#         # Read all lines at initialization
#         with open(self.file_path, 'r') as f:
#             self.lines = f.readlines()
        
#         # Store total number of frames
#         self.total_frames = len(self.lines)
        
#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         velocity_threshold: float,
#         asset_cfg: SceneEntityCfg,
#         file_path: str,
#         std: float
#     ) -> torch.Tensor:
#         """Penalize joint position error from default on the articulation."""
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#         body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#         # Get the animation joint positions and ensure they're on the same device
#         animation_joint_pos = self.__read_next_pose__(device=asset.data.joint_pos.device, NUM_INSTANCES=asset.data.joint_pos.shape[0], NUM_JOINTS=asset.data.joint_pos.shape[1])
        
#         # Only consider error in joint indices 0, 4, and 8
#         # joint_indices = [0, 4, 8]
        
#         # Calculate squared error only for the specified joints
#         error = torch.zeros_like(asset.data.joint_pos)
#         # error[:, joint_indices] = torch.square(asset.data.joint_pos[:, joint_indices] - animation_joint_pos[:, joint_indices])
#         error = torch.square(asset.data.joint_pos - animation_joint_pos)
#         # Sum error only across the specified joints
#         reward = torch.exp(-torch.sum(error, dim=1) / std)
        
#         return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
#         # return reward
    
#     def __read_next_pose__(self, device=None, NUM_INSTANCES=None, NUM_JOINTS=None):
#         """
#         Read the next pose from the file and return as a PyTorch tensor.
#         When reaching the end of file, loops back to the beginning.
#         Each pose is repeated for NUM_INSTANCES times.
        
#         Args:
#             device: The device to place the tensor on (default: None, uses self.device)
            
#         Returns:
#             torch.Tensor: Tensor of shape (NUM_INSTANCES, NUM_JOINTS) containing joint angles
#         """
#         # If no device specified, use the stored device
#         # if device is None:
#         #     device = self.device
            
#         # If we reach the end, loop back to beginning
#         if self.current_line >= self.total_frames:
#             self.current_line = 0
        
#         # print(self.current_line)

#         # Read the current line and split into values
#         line = self.lines[self.current_line]
#         values = line.strip().split()
        
#         # Convert to float and create tensor
#         joint_angles = [float(x) for x in values]
#         single_pose_tensor = torch.tensor(joint_angles, device=device).reshape(1, NUM_JOINTS)
        
#         # Repeat the pose NUM_INSTANCES times
#         # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
#         # Increment line counter
#         self.current_line += 1

#         if NUM_INSTANCES > 1:
#             # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
#             return single_pose_tensor.expand(NUM_INSTANCES, NUM_JOINTS)
#         else:
#             return single_pose_tensor
#         # return pose_tensor


class AnimationPositionReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.
    
    A class to read joint angles data from a text file and convert to PyTorch tensors.
    The data format is expected to be space-separated values with 12 joint angles per line.
    When reaching the end of file, it automatically loops back to the beginning.
    Each pose is extended to match the number of environment instances.
    
    The class now syncs with motion_sequence_counter to determine which line to read for each environment.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.file_path: str = cfg.params["file_path"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # Read all joint angles at initialization and convert to tensor immediately
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(lines)
        
        # Convert all lines to a single tensor of shape [total_frames, num_joints]
        # This is much faster than parsing each line every time
        self.joint_poses = []
        for line in lines:
            values = line.strip().split()
            joint_angles = [float(x) for x in values]
            self.joint_poses.append(joint_angles)
        
        # Convert list to tensor, will be moved to correct device during first call
        self.joint_poses_tensor = torch.tensor(self.joint_poses)
        
        # Track the device using a property instead of direct assignment
        self._current_device = None
        # Cache for the last known number of environments
        self.last_num_envs = None
        
        print(f"AnimationPositionReward initialized with {self.total_frames} frames from {self.file_path}")
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        velocity_threshold: float,
        asset_cfg: SceneEntityCfg,
        file_path: str,
        std: float
    ) -> torch.Tensor:
        """Penalize joint position error from default on the articulation."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        
        # Move tensor to correct device if needed
        current_device = asset.data.joint_pos.device
        if self._current_device != current_device:
            self._current_device = current_device
            self.joint_poses_tensor = self.joint_poses_tensor.to(current_device)
        
        # Get the animation joint positions for each environment based on motion sequence context
        animation_joint_pos = self.__get_env_poses__(
            env=env,
            NUM_ENVS=asset.data.joint_pos.shape[0], 
            NUM_JOINTS=asset.data.joint_pos.shape[1]
        )
        
        # Calculate squared error for all joints
        error = torch.square(asset.data.joint_pos - animation_joint_pos)
        
        # Calculate reward using negative exponential of error
        reward = torch.exp(-torch.sum(error, dim=1) / std)

        # reward = torch.sum(error, dim=1)
        
        # Apply velocity threshold if needed
        return torch.where(torch.logical_or(cmd > 0.0, body_vel < velocity_threshold), reward, torch.zeros_like(reward))
    
    def __get_env_poses__(self, env, NUM_ENVS=None, NUM_JOINTS=None):
        """
        Get poses for each environment based on the motion_sequence_context.
        Uses pre-loaded tensor data for efficiency.
        
        Args:
            env: The environment instance
            NUM_ENVS: Number of environments
            NUM_JOINTS: Number of joints in each pose
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_ENVS, NUM_JOINTS) containing joint angles
        """
        # Fast path: check if motion sequence context exists
        # We only need to check once if the context doesn't exist
        if not hasattr(env, "_motion_sequence_context"):
            # Use frame 0 for all environments if context doesn't exist
            return self.joint_poses_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get raw counter values from motion sequence context
        context = env._motion_sequence_context
        raw_counter_key = "raw_motion_sequence_counter"
        
        if raw_counter_key not in context:
            # Use frame 0 for all environments if counters don't exist
            return self.joint_poses_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get the current counters for all environments
        raw_counters = context[raw_counter_key]

        # print(f"Raw counters for first 3 envs: {raw_counters[:3, 0].cpu().tolist()}")
        
        # Clamp indices to valid range
        indices = raw_counters[:, 0].long().clamp(0, self.total_frames - 1)
        
        # Use advanced indexing to get all poses at once - much faster!
        # This selects the appropriate pose for each environment in a single operation
        return self.joint_poses_tensor[indices]

# class FootPositionReward(ManagerTermBase):
#     """Reward term for matching foot positions of quadrupeds.
#     A class to read target foot positions from a text file and calculate rewards
#     based on how closely the current foot positions match the targets.
#     The data format is expected to be space-separated values with 12 values per line 
#     (3 coordinates x 4 feet: FL_foot, FR_foot, RL_foot, RR_foot).
#     When reaching the end of file, it automatically loops back to the beginning.
#     """
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         """Initialize the term.
#         Args:
#             cfg: The configuration of the reward.
#             env: The RL environment instance.
#         """
#         super().__init__(cfg, env)
#         self.std: float = cfg.params["std"]
#         self.file_path: str = cfg.params["file_path"]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
#         # Define the foot link names
#         self.foot_link_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
#         # Validate that all foot links exist in the asset
#         for foot_name in self.foot_link_names:
#             if foot_name not in self.asset.body_names:
#                 raise ValueError(f"Foot link '{foot_name}' not found in asset body names")
        
#         # Get foot link indices
#         self.foot_indices = [self.asset.data.body_names.index(name) for name in self.foot_link_names]
        
#         self.current_line = 0
        
#         # Read all lines at initialization
#         with open(self.file_path, 'r') as f:
#             self.lines = f.readlines()
        
#         # Store total number of frames
#         self.total_frames = len(self.lines)

#         # Other initializations
#         self.endpoint_asset_x = []
#         self.endpoint_asset_z = []
#         self.endpoint_txt_x = []
#         self.endpoint_txt_z = []
#         self.step = 0
        
#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         asset_cfg: SceneEntityCfg,
#         file_path: str,
#         std: float
#     ) -> torch.Tensor:
#         """Calculate reward based on matching target foot positions."""
#         # Extract the asset
#         asset: Articulation = env.scene[asset_cfg.name]
#         cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        
#         # Read target foot positions from file
#         target_positions = self.__read_next_pose__(
#             device=asset.data.body_link_pos_w.device, 
#             NUM_INSTANCES=asset.data.body_link_pos_w.shape[0]
#         )
        
#         # Get root positions and orientations
#         root_position = asset.data.root_link_pos_w  # Shape: (num_instances, 3)
#         root_orientation = asset.data.root_link_quat_w  # Shape: (num_instances, 4) in (w,x,y,z) format
        
#         # Initialize total error tensor
#         total_error = torch.zeros(asset.data.body_link_pos_w.shape[0], device=asset.data.body_link_pos_w.device)
        
#         # Calculate errors for each foot
#         for i, foot_index in enumerate(self.foot_indices):
#             # if i!=0:
#             #     continue
#             # Get global foot position
#             foot_position_global = asset.data.body_link_pos_w[:, foot_index, :]  # Shape: (num_instances, 3)
            
#             # Calculate position difference in world frame
#             pos_diff_world = foot_position_global - root_position
            
#             # Rotate the position difference to root's local frame
#             foot_position_local = quat_rotate_inverse(root_orientation, pos_diff_world)

#             # Define which indices you want to extract (x and z)
#             indices = [0, 1, 2]

#             # Extract those indices from both tensors
#             target_pos_foot = target_positions[:, i*3:i*3+3][:, indices]  # Shape: [batch_size, 2]
#             foot_position_local_selected = foot_position_local[:, indices]  # Shape: [batch_size, 2]

#             # Calculate squared error and sum across the component dimension
#             foot_error = torch.sum(torch.square(foot_position_local_selected - target_pos_foot), dim=1)
            
#             # Add to total error
#             total_error += foot_error

#             # if i==0 and self.step<502:
#             #     print('append data step', self.step)
#             #     # Store both x and z coordinates
#             #     self.endpoint_asset_x.append(foot_position_local_selected[0, 0].cpu().item())  # x-coordinate
#             #     self.endpoint_asset_z.append(foot_position_local_selected[0, 1].cpu().item())  # z-coordinate
#             #     self.endpoint_txt_x.append(target_pos_foot[0, 0].cpu().item())  # x-coordinate
#             #     self.endpoint_txt_z.append(target_pos_foot[0, 1].cpu().item())  # z-coordinate
#             #     self.step += 1
#             #     if self.step > 500:
#             #         # Create a figure with two subplots
#             #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    
#             #         # Plot X coordinates
#             #         ax1.plot(self.endpoint_asset_x, label='Asset Endpoint (X)')
#             #         ax1.plot(self.endpoint_txt_x, label='Target Endpoint (X)')
#             #         ax1.set_xlabel('Steps')
#             #         ax1.set_ylabel('X Position')
#             #         ax1.legend()
#             #         ax1.set_title('X Coordinate Comparison')
                    
#             #         # Plot Z coordinates
#             #         ax2.plot(self.endpoint_asset_z, label='Asset Endpoint (Z)')
#             #         ax2.plot(self.endpoint_txt_z, label='Target Endpoint (Z)')
#             #         ax2.set_xlabel('Steps')
#             #         ax2.set_ylabel('Z Position')
#             #         ax2.legend()
#             #         ax2.set_title('Z Coordinate Comparison')
                    
#             #         plt.tight_layout()
#             #         plt.savefig('endpoint_comparison.png')
#             #         plt.close()
        
#         # Calculate reward using exponential of negative error
#         reward = torch.exp(-total_error / std)
        
#         # Return the reward directly
#         return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
#         # return reward
    
#     def __read_next_pose__(self, device=None, NUM_INSTANCES=None):
#         """
#         Read the next target foot positions from the file and return as a PyTorch tensor.
#         When reaching the end of file, loops back to the beginning.
        
#         Args:
#             device: The device to place the tensor on
#             NUM_INSTANCES: Number of instances to replicate the pose for
            
#         Returns:
#             torch.Tensor: Tensor of shape (NUM_INSTANCES, 12) containing target foot positions
#                           (3 coordinates x 4 feet)
#         """
#         # If we reach the end, loop back to beginning
#         if self.current_line >= self.total_frames:
#             self.current_line = 0
            
#         # Read the current line and split into values
#         line = self.lines[self.current_line]
#         values = line.strip().split()
        
#         # Convert to float and create tensor
#         # Expecting 12 values (3 coordinates x 4 feet)
#         if len(values) != 12:
#             raise ValueError(f"Expected 12 values per line, but got {len(values)} in line {self.current_line+1}")
            
#         target_positions = [float(x) for x in values]
#         single_pose_tensor = torch.tensor(target_positions, device=device).reshape(1, 12)
        
#         # Repeat the pose NUM_INSTANCES times
#         # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
#         # Increment line counter
#         self.current_line += 1
#         if NUM_INSTANCES > 1:
#             # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
#             return single_pose_tensor.expand(NUM_INSTANCES, 12)
#         else:
#             return single_pose_tensor
#         # return pose_tensor


class FootPositionReward(ManagerTermBase):
    """Reward term for matching foot positions of quadrupeds.
    
    A class to read target foot positions from a text file and calculate rewards
    based on how closely the current foot positions match the targets.
    The data format is expected to be space-separated values with 12 values per line 
    (3 coordinates x 4 feet: FL_foot, FR_foot, RL_foot, RR_foot).
    
    The class now syncs with motion_sequence_counter to determine which line to read for each environment.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.file_path: str = cfg.params["file_path"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # Define the foot link names
        self.foot_link_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        # Validate that all foot links exist in the asset
        for foot_name in self.foot_link_names:
            if foot_name not in self.asset.body_names:
                raise ValueError(f"Foot link '{foot_name}' not found in asset body names")
        
        # Get foot link indices
        self.foot_indices = [self.asset.data.body_names.index(name) for name in self.foot_link_names]
        
        # Read all target positions at initialization and convert to tensor immediately
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(lines)
        
        # Convert all lines to a single tensor of shape [total_frames, 12]
        # This is much faster than parsing each line every time
        self.target_positions = []
        for line in lines:
            values = line.strip().split()
            if len(values) != 12:
                print(f"Warning: Expected 12 values but got {len(values)}. Using zeros.")
                position_values = [0.0] * 12
            else:
                position_values = [float(x) for x in values]
            self.target_positions.append(position_values)
        
        # Convert list to tensor, will be moved to correct device during first call
        self.target_positions_tensor = torch.tensor(self.target_positions)
        
        # Track the device using a property
        self._current_device = None
        
        print(f"FootPositionReward initialized with {self.total_frames} frames from {self.file_path}")
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        file_path: str,
        std: float
    ) -> torch.Tensor:
        """Calculate reward based on matching target foot positions."""
        # Extract the asset
        asset: Articulation = env.scene[asset_cfg.name]
        cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        
        # Move tensor to correct device if needed
        current_device = asset.data.body_link_pos_w.device
        if self._current_device != current_device:
            self._current_device = current_device
            self.target_positions_tensor = self.target_positions_tensor.to(current_device)
        
        # Get target foot positions based on motion sequence context
        target_positions = self.__get_target_positions__(
            env=env,
            NUM_ENVS=asset.data.body_link_pos_w.shape[0]
        )
        
        # Get root positions and orientations
        root_position = asset.data.root_link_pos_w  # Shape: (num_instances, 3)
        root_orientation = asset.data.root_link_quat_w  # Shape: (num_instances, 4) in (w,x,y,z) format
        
        # Initialize total error tensor
        total_error = torch.zeros(asset.data.body_link_pos_w.shape[0], device=current_device)
        
        # Calculate errors for each foot
        for i, foot_index in enumerate(self.foot_indices):
            # Get global foot position
            foot_position_global = asset.data.body_link_pos_w[:, foot_index, :]  # Shape: (num_instances, 3)
            
            # Calculate position difference in world frame
            pos_diff_world = foot_position_global - root_position
            
            # Rotate the position difference to root's local frame
            foot_position_local = quat_rotate_inverse(root_orientation, pos_diff_world)

            # Define which indices you want to extract (x, y, and z)
            indices = [0, 1, 2]

            # Extract those indices from both tensors
            target_pos_foot = target_positions[:, i*3:i*3+3][:, indices]  # Shape: [batch_size, 3]
            foot_position_local_selected = foot_position_local[:, indices]  # Shape: [batch_size, 3]

            # Calculate squared error and sum across the component dimension
            foot_error = torch.sum(torch.square(foot_position_local_selected - target_pos_foot), dim=1)
            
            # Add to total error
            total_error += foot_error
        
        # Calculate reward using exponential of negative error
        reward = torch.exp(-total_error / std)

        # reward = total_error
        
        # Return the reward with velocity condition applied
        return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
    
    def __get_target_positions__(self, env, NUM_ENVS=None):
        """
        Get target positions for each environment based on the motion_sequence_context.
        Uses pre-loaded tensor data for efficiency.
        
        Args:
            env: The environment instance
            NUM_ENVS: Number of environments
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_ENVS, 12) containing target foot positions
        """
        # Fast path: check if motion sequence context exists
        if not hasattr(env, "_motion_sequence_context"):
            # Use frame 0 for all environments if context doesn't exist
            return self.target_positions_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get raw counter values from motion sequence context
        context = env._motion_sequence_context
        raw_counter_key = "raw_motion_sequence_counter"
        
        if raw_counter_key not in context:
            # Use frame 0 for all environments if counters don't exist
            return self.target_positions_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get the current counters for all environments
        raw_counters = context[raw_counter_key]
        
        # Clamp indices to valid range
        indices = raw_counters[:, 0].long().clamp(0, self.total_frames - 1)
        
        # Use advanced indexing to get all target positions at once - much faster!
        # This selects the appropriate position for each environment in a single operation
        return self.target_positions_tensor[indices]

# class AnimationVelocityReward(ManagerTermBase):
#     """Gait enforcing reward term for quadrupeds.
#     A class to read joint angles data from a text file and convert to PyTorch tensors.
#     The data format is expected to be space-separated values with 12 joint angles per line.
#     When reaching the end of file, it automatically loops back to the beginning.
#     Each pose is extended to 4096 identical instances.
#     """
#     def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
#         """Initialize the term.
#         Args:
#             cfg: The configuration of the reward.
#             env: The RL environment instance.
#             file_path (str): Path to the joint angles text file
#         """
#         super().__init__(cfg, env)
#         self.std: float = cfg.params["std"]
#         self.velocity_threshold: float = cfg.params["velocity_threshold"]
#         self.file_path: str = cfg.params["file_path"]
#         self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
#         self.current_line = 0
        
#         # Store the device from the asset for later use
#         # self.device = self.asset.data.joint_pos.device
        
#         # Read all lines at initialization
#         with open(self.file_path, 'r') as f:
#             self.lines = f.readlines()
        
#         # Store total number of frames
#         self.total_frames = len(self.lines)
        
#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         velocity_threshold: float,
#         asset_cfg: SceneEntityCfg,
#         file_path: str,
#         std: float
#     ) -> torch.Tensor:
#         """Penalize joint position error from default on the articulation."""
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
#         body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
#         # Get the animation joint positions and ensure they're on the same device
#         animation_joint_vel = self.__read_next_pose__(device=asset.data.joint_vel.device, NUM_INSTANCES=asset.data.joint_vel.shape[0], NUM_JOINTS=asset.data.joint_vel.shape[1])
        
#         # Only consider error in joint indices 0, 4, and 8
#         # joint_indices = [0, 4, 8]
        
#         # Calculate squared error only for the specified joints
#         error = torch.zeros_like(asset.data.joint_vel)
#         # error[:, joint_indices] = torch.square(asset.data.joint_pos[:, joint_indices] - animation_joint_pos[:, joint_indices])
#         error = torch.square(asset.data.joint_vel - animation_joint_vel)
#         # Sum error only across the specified joints
#         reward = torch.exp(-torch.sum(error, dim=1) / std)
        
#         return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
#         # return reward
    
#     def __read_next_pose__(self, device=None, NUM_INSTANCES=None, NUM_JOINTS=None):
#         """
#         Read the next pose from the file and return as a PyTorch tensor.
#         When reaching the end of file, loops back to the beginning.
#         Each pose is repeated for NUM_INSTANCES times.
        
#         Args:
#             device: The device to place the tensor on (default: None, uses self.device)
            
#         Returns:
#             torch.Tensor: Tensor of shape (NUM_INSTANCES, NUM_JOINTS) containing joint angles
#         """
#         # If no device specified, use the stored device
#         # if device is None:
#         #     device = self.device
            
#         # If we reach the end, loop back to beginning
#         if self.current_line >= self.total_frames:
#             self.current_line = 0
            
#         # Read the current line and split into values
#         line = self.lines[self.current_line]
#         values = line.strip().split()
        
#         # Convert to float and create tensor
#         joint_vels = [float(x) for x in values]
#         single_pose_tensor = torch.tensor(joint_vels, device=device).reshape(1, NUM_JOINTS)
        
#         # Repeat the pose NUM_INSTANCES times
#         # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
#         # Increment line counter
#         self.current_line += 1
#         if NUM_INSTANCES > 1:
#             # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
#             return single_pose_tensor.expand(NUM_INSTANCES, NUM_JOINTS)
#         else:
#             return single_pose_tensor
#         # return pose_tensor

class AnimationVelocityReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.
    
    A class to read joint velocity data from a text file and convert to PyTorch tensors.
    The data format is expected to be space-separated values with joint velocities per line.
    When reaching the end of file, it automatically loops back to the beginning.
    Each pose is extended to match the number of environment instances.
    
    The class now syncs with motion_sequence_counter to determine which line to read for each environment.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
            file_path (str): Path to the joint velocities text file
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.file_path: str = cfg.params["file_path"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # Read all joint velocities at initialization and convert to tensor immediately
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(lines)
        
        # Convert all lines to a single tensor of shape [total_frames, num_joints]
        # This is much faster than parsing each line every time
        self.joint_vels = []
        num_joints = None
        
        for line in lines:
            values = line.strip().split()
            
            # Set expected number of joints from first line
            if num_joints is None:
                num_joints = len(values)
                
            # Check if current line has expected number of values
            if len(values) != num_joints:
                print(f"Warning: Expected {num_joints} values but got {len(values)}. Using zeros.")
                joint_velocities = [0.0] * num_joints
            else:
                joint_velocities = [float(x) for x in values]
                
            self.joint_vels.append(joint_velocities)
        
        # Convert list to tensor, will be moved to correct device during first call
        self.joint_vels_tensor = torch.tensor(self.joint_vels)
        
        # Track the device using a property
        self._current_device = None
        
        print(f"AnimationVelocityReward initialized with {self.total_frames} frames from {self.file_path}")
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        velocity_threshold: float,
        asset_cfg: SceneEntityCfg,
        file_path: str,
        std: float
    ) -> torch.Tensor:
        """Penalize joint velocity error from the target animation."""
        # Extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        
        # Move tensor to correct device if needed
        current_device = asset.data.joint_vel.device
        if self._current_device != current_device:
            self._current_device = current_device
            self.joint_vels_tensor = self.joint_vels_tensor.to(current_device)
        
        # Get the animation joint velocities for each environment based on motion sequence context
        animation_joint_vel = self.__get_joint_velocities__(
            env=env,
            NUM_ENVS=asset.data.joint_vel.shape[0], 
            NUM_JOINTS=asset.data.joint_vel.shape[1]
        )
        
        # Calculate squared error between actual and target joint velocities
        error = torch.square(asset.data.joint_vel - animation_joint_vel)
        
        # Calculate reward using negative exponential of error
        reward = torch.exp(-torch.sum(error, dim=1) / std)
        
        # reward = torch.sum(error, dim=1)
        
        # Return the reward with velocity condition applied
        return torch.where(torch.logical_or(cmd > 0.0, body_vel < velocity_threshold), reward, torch.zeros_like(reward))
    
    def __get_joint_velocities__(self, env, NUM_ENVS=None, NUM_JOINTS=None):
        """
        Get joint velocities for each environment based on the motion_sequence_context.
        Uses pre-loaded tensor data for efficiency.
        
        Args:
            env: The environment instance
            NUM_ENVS: Number of environments
            NUM_JOINTS: Number of joints in each pose
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_ENVS, NUM_JOINTS) containing joint velocities
        """
        # Check tensor dimensions and reshape if needed
        if self.joint_vels_tensor.shape[1] != NUM_JOINTS:
            print(f"Warning: Joint velocity tensor has {self.joint_vels_tensor.shape[1]} joints, but asset has {NUM_JOINTS}.")
            # If we have fewer joints than expected, pad with zeros
            if self.joint_vels_tensor.shape[1] < NUM_JOINTS:
                padding = torch.zeros((self.total_frames, NUM_JOINTS - self.joint_vels_tensor.shape[1]), 
                                     device=self._current_device)
                padded_tensor = torch.cat([self.joint_vels_tensor, padding], dim=1)
                return padded_tensor[0].repeat(NUM_ENVS, 1)
            # If we have more joints than expected, truncate
            else:
                truncated_tensor = self.joint_vels_tensor[:, :NUM_JOINTS]
                return truncated_tensor[0].repeat(NUM_ENVS, 1)
        
        # Fast path: check if motion sequence context exists
        if not hasattr(env, "_motion_sequence_context"):
            # Use frame 0 for all environments if context doesn't exist
            return self.joint_vels_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get raw counter values from motion sequence context
        context = env._motion_sequence_context
        raw_counter_key = "raw_motion_sequence_counter"
        
        if raw_counter_key not in context:
            # Use frame 0 for all environments if counters don't exist
            return self.joint_vels_tensor[0].repeat(NUM_ENVS, 1)
        
        # Get the current counters for all environments
        raw_counters = context[raw_counter_key]
        
        # Clamp indices to valid range
        indices = raw_counters[:, 0].long().clamp(0, self.total_frames - 1)
        
        # Use advanced indexing to get all joint velocities at once - much faster!
        # This selects the appropriate velocities for each environment in a single operation
        return self.joint_vels_tensor[indices]

class ContactPhaseReward(ManagerTermBase):
    """Reward that checks if foot contacts match the expected gait pattern.
    
    Uses the motion_sequence_counter to determine which contact phase should be active,
    and rewards environments based on how many feet match the expected contact pattern.
    The reward ranges from 0 to 4, representing the count of correctly matched contacts.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        
        # Extract parameters from configuration
        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.threshold = cfg.params["threshold"]
        self.contact_phases = cfg.params["contact_phases"]
        self.total_phases = len(self.contact_phases)
        
        # Pre-compute the expected contacts for each phase and convert to tensor
        # This avoids string parsing during the reward calculation
        self.expected_contacts_tensor = torch.zeros((self.total_phases, 4), dtype=torch.bool)
        for i, phase in enumerate(self.contact_phases):
            for j, state in enumerate(phase):
                self.expected_contacts_tensor[i, j] = (state == '1')
        
        # Store device reference
        self._device = None
        
        print(f"ContactPhaseReward initialized with {self.total_phases} contact phases")
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        threshold: float,
        contact_phases: List[str]
    ) -> torch.Tensor:
        """Calculate reward based on matching foot contacts with the expected pattern.
        
        Args:
            env: The environment.
            asset_cfg: Configuration for the robot asset with foot bodies.
            sensor_cfg: Configuration for the contact sensor.
            threshold: Threshold to determine if a foot is in contact.
            contact_phases: List of contact phase patterns as strings (e.g., '1010').
            
        Returns:
            Reward tensor with shape (num_envs,) where each value is between 0 and 4
            representing the sum of correctly matched foot contacts.
        """
        # Extract the asset to get the number of environments and device
        asset: Articulation = env.scene[asset_cfg.name]
        num_envs = asset.data.joint_pos.shape[0]
        device = asset.data.joint_pos.device
        
        # Move expected contacts tensor to the correct device if needed
        if self._device != device:
            self._device = device
            self.expected_contacts_tensor = self.expected_contacts_tensor.to(device)
        
        # Get the current motion sequence counter for each environment
        # Fast path: Use motion_sequence_context if available
        if hasattr(env, "_motion_sequence_context"):
            context = env._motion_sequence_context
            raw_counter_key = "raw_motion_sequence_counter"
            
            if raw_counter_key in context:
                phase_indices = context[raw_counter_key][:, 0].long() % self.total_phases
            else:
                # Fallback if no counter exists
                phase_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        else:
            # Check if motion_sequence_counter exists in env.extras
            counter_key = "motion_sequence_counter"
            
            if counter_key in env.extras:
                phase_indices = env.extras[counter_key][:, 0].long() % self.total_phases
            else:
                # Fallback if no counter exists
                phase_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Get contact sensor and check contacts efficiently
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history
        
        # Efficiently check contacts for all feet at once
        # Assume body_ids are ordered as [FL, FR, RL, RR]
        body_ids = sensor_cfg.body_ids
        num_feet = 4  # Assuming exactly 4 feet for a quadruped
        
        # Pre-allocate actual contacts tensor
        actual_contacts = torch.zeros((num_envs, num_feet), device=device, dtype=torch.bool)
        
        # Vectorized contact detection for performance
        for i in range(num_feet):
            foot_id = body_ids[i] if isinstance(body_ids, list) else i
            
            # Max force over time dimension (dim=1) for this foot
            max_forces = torch.max(torch.norm(net_contact_forces[:, :, foot_id], dim=-1), dim=1)[0]
            actual_contacts[:, i] = max_forces > threshold
        
        # Select the expected contacts for each environment based on its phase index
        # This is much faster than looping through environments
        expected_contacts = self.expected_contacts_tensor[phase_indices]
        
        # Calculate matches between expected and actual contacts
        # Instead of requiring all contacts to match, count how many match
        # This creates a tensor of shape (num_envs, num_feet) with 1.0 where contacts match
        contact_matches = (actual_contacts == expected_contacts).float()
        
        # Sum the matches for each environment without normalizing
        # Result shape: (num_envs,)
        return torch.sum(contact_matches, dim=1) / num_feet