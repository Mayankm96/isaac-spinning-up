# This file includes code derived from HybridRobotics/whole_body_tracking
# (MIT License).
#
# Original source:
# https://github.com/HybridRobotics/whole_body_tracking
#
# Modifications:
#   - Added docstrings.
#   - Added type hints.
#
# See docs/licenses/LICENSE-whole_body_tracking.txt for the full license text.

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from motion_tracking.dataset.motion_loader import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .motion_command_cfg import MotionCommandCfg


class MotionCommand(CommandTerm):
    """Command term for the motion tracking task.

    This command term loads the motion data from a file and provides the reference motion
    for the task. It also logs the metrics for the motion tracking task.

    Training a motion tracking policy is challenging due to the long-horizon nature of the task.
    Not all segments are equally difficult. Instead of uniform sampling the starting index of
    the motion, an adaptive sampling strategy is used.

    To achieve this, we divide the starting index of the entire motion into S bins of one second each and
    sample these bins according to empirical failure statistics. Let Ns and Fs be the counts of episodes
    and failures starting in bin s. The failure rate is smoothed over time using an exponential moving
    average to prevent discrete jumps in sampling.

    For more details, please refer to the paper: https://arxiv.org/abs/2508.08241
    """

    cfg: MotionCommandCfg
    """Configuration for the motion command."""

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # extract the indices of the anchor and motion bodies
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indices = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # load the motion data
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indices, device=self.device)

        # initialize the internal variables
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # initialize the adaptive sampling variables
        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # initialize the metrics
        self.metrics = {
            "error_anchor_pos": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_rot": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_lin_vel": torch.zeros(self.num_envs, device=self.device),
            "error_anchor_ang_vel": torch.zeros(self.num_envs, device=self.device),
            "error_body_pos": torch.zeros(self.num_envs, device=self.device),
            "error_body_rot": torch.zeros(self.num_envs, device=self.device),
            "error_joint_pos": torch.zeros(self.num_envs, device=self.device),
            "error_joint_vel": torch.zeros(self.num_envs, device=self.device),
            "sampling_entropy": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_prob": torch.zeros(self.num_envs, device=self.device),
            "sampling_top1_bin": torch.zeros(self.num_envs, device=self.device),
        }

    def __str__(self) -> str:
        msg = "MotionCommand:\n"
        msg += f"\tAsset name: {self.cfg.asset_name}\n"
        msg += f"\tMotion file: {self.cfg.motion_file}\n"
        msg += f"\tAnchor body name: {self.cfg.anchor_body_name}\n"
        msg += f"\tBody names: {self.cfg.body_names}"
        return msg

    """
    Properties
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command has a debug visualization implementation."""
        return "dummy_robot" in self._env.scene.keys()

    @property
    def command(self) -> torch.Tensor:
        """The command tensor. Shape is (num_envs, 2 * num_joints)."""
        # note: we do not use this command tensor directly.
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    """
    Properties - Data Loader Access.
    """

    @property
    def joint_pos(self) -> torch.Tensor:
        """The desired joint positions. Shape is (num_envs, num_joints)."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """The desired joint velocities. Shape is (num_envs, num_joints)."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """The desired body positions in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """The desired body quaternions in the world frame. Shape is (num_envs, num_bodies, 4).

        The quaternion is in the format of (w, x, y, z).
        """
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """The desired body linear velocities in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """The desired body angular velocities in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """The desired anchor position in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """The desired anchor quaternion in the world frame. Shape is (num_envs, 4).

        The quaternion is in the format of (w, x, y, z).
        """
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """The desired anchor linear velocity in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """The desired anchor angular velocity in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    """
    Properties - Robot Access.
    """

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """The current joint positions of the robot. Shape is (num_envs, num_joints)."""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """The current joint velocities of the robot. Shape is (num_envs, num_joints)."""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """The current body positions of the robot in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_pos_w[:, self.body_indices]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """The current body quaternions of the robot in the world frame. Shape is (num_envs, num_bodies, 4).

        The quaternion is in the format of (w, x, y, z).
        """
        return self.robot.data.body_quat_w[:, self.body_indices]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """The current body linear velocities of the robot in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_lin_vel_w[:, self.body_indices]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """The current body angular velocities of the robot in the world frame. Shape is (num_envs, num_bodies, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_ang_vel_w[:, self.body_indices]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """The current anchor position of the robot in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """The current anchor quaternion of the robot in the world frame. Shape is (num_envs, 4).

        The quaternion is in the format of (w, x, y, z).
        """
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """The current anchor linear velocity of the robot in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """The current anchor angular velocity of the robot in the world frame. Shape is (num_envs, 3).

        Note: World frame refers to the simulation world frame and not environment frame.
        """
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the given environment indices."""
        # check if there are any environment indices to resample
        if len(env_ids) == 0:
            return

        # adaptive sampling to compute the starting index of the motion
        self._adaptive_sampling(env_ids)

        # sample the root pose and velocity from the ranges
        # -- obtain the reference root pose and velocity from the motion
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # -- draw random samples from the ranges
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        # -- position
        root_pos[env_ids] += rand_samples[:, 0:3]
        # -- orientation
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # -- draw random samples from the ranges
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        # -- velocity
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # sample the joint positions and velocities from the ranges
        # -- obtain the reference joint positions and velocities from the motion
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        # -- draw random samples from the ranges
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        # write the root and joint states to the simulation
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update the command by incrementing the time steps and resampling the command if needed."""
        # increment the time steps
        self.time_steps += 1

        # identify the environment indices that have reached the end of the motion
        # for these we resample the command to not have the robot just stay in the same position.
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        # repeat the anchor and robot anchor positions and quaternions to match the body names
        # all tensors have shape (num_envs, dims) --> (num_envs, num_bodies, dims)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        # compute the control frame of the robot in the world frame
        # control frame has only the xy position and yaw orientation.
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        # compute the pose of all the bodies in the control frame
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # update the failed bins
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    """
    Debug visualization functions.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # obtain the dummy robot and door for visualization
        if not hasattr(self, "_dummy_robot"):
            if "dummy_robot" not in self._env.scene.keys():
                omni.log.warn("Dummy robot not found in the scene. No debug visualization available.")
                return
            # set internal variables
            self._dummy_robot: Articulation = self._env.scene["dummy_robot"]

        # set visibility of markers
        self._dummy_robot.set_visibility(debug_vis, self._dummy_robot._ALL_INDICES)  # type: ignore

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if hasattr(self, "_dummy_robot") and self._dummy_robot.is_initialized:
            # -- dummy robot
            self._dummy_robot.write_root_state_to_sim(
                torch.cat(
                    [self.anchor_pos_w, self.anchor_quat_w, self.anchor_lin_vel_w, self.anchor_ang_vel_w], dim=-1
                ),
            )
            self._dummy_robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)

    """
    Internal helpers.
    """

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Adaptive sampling to compute the starting index of the motion."""
        # check if any episodes have failed
        episode_failed = self._env.termination_manager.terminated[env_ids]

        # update the failed bins
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
