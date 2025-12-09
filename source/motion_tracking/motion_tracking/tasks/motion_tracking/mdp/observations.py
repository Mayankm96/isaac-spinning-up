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

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Orientation of the robot anchor in the world frame. Shape is (num_envs, 6).

    The returned orientation is in the 6-DoF representation of SO(3).
    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the orientation as a 6-DoF representation of SO(3)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Linear velocity of the robot anchor in the world frame. Shape is (num_envs, 3)."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Angular velocity of the robot anchor in the world frame. Shape is (num_envs, 3)."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Position of the robot bodies in the anchor frame. Shape is (num_envs, num_bodies * 3)."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Orientation of the robot bodies in the anchor frame. Shape is (num_envs, num_bodies * 6).

    The returned orientation is in the 6-DoF representation of SO(3).
    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    # compute the orientation as a 6-DoF representation of SO(3)
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Desired position of the robot anchor in the current anchor frame. Shape is (num_envs, 3)."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Desired orientation of the robot anchor in the current anchor frame. Shape is (num_envs, 6).

    The returned orientation is in the 6-DoF representation of SO(3).
    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the orientation as a 6-DoF representation of SO(3)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    # compute the orientation as a 6-DoF representation of SO(3)
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
