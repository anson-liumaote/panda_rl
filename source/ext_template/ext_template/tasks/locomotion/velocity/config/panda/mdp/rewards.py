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

class AnimationReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.
    A class to read joint angles data from a text file and convert to PyTorch tensors.
    The data format is expected to be space-separated values with 12 joint angles per line.
    When reaching the end of file, it automatically loops back to the beginning.
    Each pose is extended to 4096 identical instances.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
            file_path (str): Path to the joint angles text file
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.file_path: str = cfg.params["file_path"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        self.current_line = 0
        
        # Store the device from the asset for later use
        # self.device = self.asset.data.joint_pos.device
        
        # Read all lines at initialization
        with open(self.file_path, 'r') as f:
            self.lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(self.lines)
        
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
        # Get the animation joint positions and ensure they're on the same device
        animation_joint_pos = self.__read_next_pose__(device=asset.data.joint_pos.device, NUM_INSTANCES=asset.data.joint_pos.shape[0], NUM_JOINTS=asset.data.joint_pos.shape[1])
        
        # Only consider error in joint indices 0, 4, and 8
        # joint_indices = [0, 4, 8]
        
        # Calculate squared error only for the specified joints
        error = torch.zeros_like(asset.data.joint_pos)
        # error[:, joint_indices] = torch.square(asset.data.joint_pos[:, joint_indices] - animation_joint_pos[:, joint_indices])
        error = torch.square(asset.data.joint_pos - animation_joint_pos)
        # Sum error only across the specified joints
        reward = torch.exp(-torch.sum(error, dim=1) / std)
        
        return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
        # return reward
    
    def __read_next_pose__(self, device=None, NUM_INSTANCES=None, NUM_JOINTS=None):
        """
        Read the next pose from the file and return as a PyTorch tensor.
        When reaching the end of file, loops back to the beginning.
        Each pose is repeated for NUM_INSTANCES times.
        
        Args:
            device: The device to place the tensor on (default: None, uses self.device)
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_INSTANCES, NUM_JOINTS) containing joint angles
        """
        # If no device specified, use the stored device
        # if device is None:
        #     device = self.device
            
        # If we reach the end, loop back to beginning
        if self.current_line >= self.total_frames:
            self.current_line = 0
        
        # print(self.current_line)

        # Read the current line and split into values
        line = self.lines[self.current_line]
        values = line.strip().split()
        
        # Convert to float and create tensor
        joint_angles = [float(x) for x in values]
        single_pose_tensor = torch.tensor(joint_angles, device=device).reshape(1, NUM_JOINTS)
        
        # Repeat the pose NUM_INSTANCES times
        # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
        # Increment line counter
        self.current_line += 1
        

        if NUM_INSTANCES > 1:
            # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
            return single_pose_tensor.expand(NUM_INSTANCES, NUM_JOINTS)
        else:
            return single_pose_tensor
        # return pose_tensor

class FootPositionReward(ManagerTermBase):
    """Reward term for matching foot positions of quadrupeds.
    A class to read target foot positions from a text file and calculate rewards
    based on how closely the current foot positions match the targets.
    The data format is expected to be space-separated values with 12 values per line 
    (3 coordinates x 4 feet: FL_foot, FR_foot, RL_foot, RR_foot).
    When reaching the end of file, it automatically loops back to the beginning.
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
        
        self.current_line = 0
        
        # Read all lines at initialization
        with open(self.file_path, 'r') as f:
            self.lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(self.lines)

        self.step = 0
        self.endpoint_txt = []
        self.endpoint_asset = []
        
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
        
        # Read target foot positions from file
        target_positions = self.__read_next_pose__(
            device=asset.data.body_link_pos_w.device, 
            NUM_INSTANCES=asset.data.body_link_pos_w.shape[0]
        )
        
        # Get root positions and orientations
        root_position = asset.data.root_link_pos_w  # Shape: (num_instances, 3)
        root_orientation = asset.data.root_link_quat_w  # Shape: (num_instances, 4) in (w,x,y,z) format
        
        # Initialize total error tensor
        total_error = torch.zeros(asset.data.body_link_pos_w.shape[0], device=asset.data.body_link_pos_w.device)
        
        # Calculate errors for each foot
        for i, foot_index in enumerate(self.foot_indices):
            # if i!=0:
            #     continue
            # Get global foot position
            foot_position_global = asset.data.body_link_pos_w[:, foot_index, :]  # Shape: (num_instances, 3)
            
            # Calculate position difference in world frame
            pos_diff_world = foot_position_global - root_position
            
            # Rotate the position difference to root's local frame
            foot_position_local = quat_rotate_inverse(root_orientation, pos_diff_world)

            # Define which indices you want to extract (x and z)
            indices = [0, 2]

            # Extract those indices from both tensors
            target_pos_foot = target_positions[:, i*3:i*3+3][:, indices]  # Shape: [batch_size, 2]
            foot_position_local_selected = foot_position_local[:, indices]  # Shape: [batch_size, 2]

            # Calculate squared error and sum across the component dimension
            foot_error = torch.sum(torch.square(foot_position_local_selected - target_pos_foot), dim=1)
            
            # Add to total error
            total_error += foot_error

            # if i==0 and self.step<502:
            #     print('append data step', self.step)
            #     self.endpoint_asset.append(foot_position_local[0, 2].cpu().item())
            #     self.endpoint_txt.append(target_pos_foot[0, 2].cpu().item())
            #     self.step += 1
            #     if self.step > 500:
            #         plt.figure(figsize=(10, 6))
            #         plt.plot(self.endpoint_asset, label='Asset Endpoint')
            #         plt.plot(self.endpoint_txt, label='Target Endpoint')
            #         plt.xlabel('Steps')
            #         plt.ylabel('Z Position')
            #         plt.legend()
            #         plt.savefig('endpoint_comparison.png')
            #         plt.close()
        
        # Calculate reward using exponential of negative error
        reward = torch.exp(-total_error / std)
        
        # Return the reward directly
        return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
        # return reward
    
    def __read_next_pose__(self, device=None, NUM_INSTANCES=None):
        """
        Read the next target foot positions from the file and return as a PyTorch tensor.
        When reaching the end of file, loops back to the beginning.
        
        Args:
            device: The device to place the tensor on
            NUM_INSTANCES: Number of instances to replicate the pose for
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_INSTANCES, 12) containing target foot positions
                          (3 coordinates x 4 feet)
        """
        # If we reach the end, loop back to beginning
        if self.current_line >= self.total_frames:
            self.current_line = 0
            
        # Read the current line and split into values
        line = self.lines[self.current_line]
        values = line.strip().split()
        
        # Convert to float and create tensor
        # Expecting 12 values (3 coordinates x 4 feet)
        if len(values) != 12:
            raise ValueError(f"Expected 12 values per line, but got {len(values)} in line {self.current_line+1}")
            
        target_positions = [float(x) for x in values]
        single_pose_tensor = torch.tensor(target_positions, device=device).reshape(1, 12)
        
        # Repeat the pose NUM_INSTANCES times
        # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
        # Increment line counter
        self.current_line += 1
        if NUM_INSTANCES > 1:
            # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
            return single_pose_tensor.expand(NUM_INSTANCES, 12)
        else:
            return single_pose_tensor
        # return pose_tensor

class AnimationVelocityReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.
    A class to read joint angles data from a text file and convert to PyTorch tensors.
    The data format is expected to be space-separated values with 12 joint angles per line.
    When reaching the end of file, it automatically loops back to the beginning.
    Each pose is extended to 4096 identical instances.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.
        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
            file_path (str): Path to the joint angles text file
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.file_path: str = cfg.params["file_path"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        self.current_line = 0
        
        # Store the device from the asset for later use
        # self.device = self.asset.data.joint_pos.device
        
        # Read all lines at initialization
        with open(self.file_path, 'r') as f:
            self.lines = f.readlines()
        
        # Store total number of frames
        self.total_frames = len(self.lines)
        
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
        # Get the animation joint positions and ensure they're on the same device
        animation_joint_vel = self.__read_next_pose__(device=asset.data.joint_vel.device, NUM_INSTANCES=asset.data.joint_vel.shape[0], NUM_JOINTS=asset.data.joint_vel.shape[1])
        
        # Only consider error in joint indices 0, 4, and 8
        # joint_indices = [0, 4, 8]
        
        # Calculate squared error only for the specified joints
        error = torch.zeros_like(asset.data.joint_vel)
        # error[:, joint_indices] = torch.square(asset.data.joint_pos[:, joint_indices] - animation_joint_pos[:, joint_indices])
        error = torch.square(asset.data.joint_vel - animation_joint_vel)
        # Sum error only across the specified joints
        reward = torch.exp(-torch.sum(error, dim=1) / std)
        
        return torch.where(torch.logical_or(cmd > 0.0, cmd < 0.3), reward, 0.0)
        # return reward
    
    def __read_next_pose__(self, device=None, NUM_INSTANCES=None, NUM_JOINTS=None):
        """
        Read the next pose from the file and return as a PyTorch tensor.
        When reaching the end of file, loops back to the beginning.
        Each pose is repeated for NUM_INSTANCES times.
        
        Args:
            device: The device to place the tensor on (default: None, uses self.device)
            
        Returns:
            torch.Tensor: Tensor of shape (NUM_INSTANCES, NUM_JOINTS) containing joint angles
        """
        # If no device specified, use the stored device
        # if device is None:
        #     device = self.device
            
        # If we reach the end, loop back to beginning
        if self.current_line >= self.total_frames:
            self.current_line = 0
            
        # Read the current line and split into values
        line = self.lines[self.current_line]
        values = line.strip().split()
        
        # Convert to float and create tensor
        joint_vels = [float(x) for x in values]
        single_pose_tensor = torch.tensor(joint_vels, device=device).reshape(1, NUM_JOINTS)
        
        # Repeat the pose NUM_INSTANCES times
        # pose_tensor = single_pose_tensor.repeat(NUM_INSTANCES, 1)
        
        # Increment line counter
        self.current_line += 1
        if NUM_INSTANCES > 1:
            # For operations that need exact dimensions, we can expand (doesn't allocate new memory)
            return single_pose_tensor.expand(NUM_INSTANCES, NUM_JOINTS)
        else:
            return single_pose_tensor
        # return pose_tensor

