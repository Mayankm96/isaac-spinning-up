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

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import MotionCommand
from .utils import get_body_indices


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Check if the anchor position is beyond the threshold."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the error and check if it exceeds the threshold
    # TODO: compute position error between robot and reference
    # Hint: subtract and check if any of the errors exceed the threshold
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Check if the anchor position is beyond the threshold in the z-axis."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the error and check if it exceeds the threshold
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Check if the anchor orientation is beyond the threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the projected gravity vector for the motion and the robot
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    # compute the error and check if it exceeds the threshold
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Check if the body position is beyond the threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the body indices
    body_indices = get_body_indices(command, body_names)

    # compute the error and check if it exceeds the threshold
    error = torch.norm(
        command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices],
        dim=-1,
    )

    # check if any of the errors exceed the threshold
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Check if the body position is beyond the threshold in the z-axis."""
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the body indices
    body_indices = get_body_indices(command, body_names)

    # compute the error and check if it exceeds the threshold
    error = torch.abs(command.body_pos_relative_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1])
    return torch.any(error > threshold, dim=-1)


def base_ang_vel_exceed(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Check if the base angular velocity exceeds the threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]

    # check if any of the errors exceed the threshold
    # TODO: check if the base angular velocity exceeds the threshold
    # Hint: use the root_ang_vel_b property of the robot articulation
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
