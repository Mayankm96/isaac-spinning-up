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

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand
from .utils import get_body_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    r"""Exponential kernel reward for the global anchor position error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \|x - x_{ref}\|^2 \right)

    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # extract quantities for convenience
    anchor_pos_w = command.anchor_pos_w
    robot_anchor_pos_w = command.robot_anchor_pos_w

    # compute the error
    # TODO: compute position error between robot and reference
    # Hint: subtract and square, then sum over xyz

    return 0.0


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    r"""Exponential kernel reward for the global anchor orientation error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \|q \boxminus q_{ref}\|^2 \right)

    where :math:`\boxminus` is the quaternion box-minus operator.
    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)

    # extract quantities for convenience
    anchor_quat_w = command.anchor_quat_w
    robot_anchor_quat_w = command.robot_anchor_quat_w

    # compute the error
    # TODO: compute orientation error between robot and reference
    # Hint: use quat_error_magnitude

    return 0.0


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    r"""Exponential kernel reward for the relative body position error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \sum_{i=1}^N (x_i - x_{i,ref})^2 \right)

    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = get_body_indices(command, body_names)

    # extract quantities for convenience
    body_pos_w = command.body_pos_relative_w[:, body_indices]
    robot_body_pos_w = command.robot_body_pos_w[:, body_indices]

    # compute the error
    # TODO: compute position error between robot and reference
    # Hint: subtract and square, then sum over xyz 
    return 0.0


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    r"""Exponential kernel reward for the relative body orientation error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \sum_{i=1}^N (q_i \boxminus q_{i,ref})^2 \right)

    where :math:`\boxminus` is the quaternion box-minus operator.
    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = get_body_indices(command, body_names)

    # extract quantities for convenience
    body_quat_w = command.body_quat_relative_w[:, body_indices]
    robot_body_quat_w = command.robot_body_quat_w[:, body_indices]

    # compute the error
    # TODO: compute orientation error between robot and reference
    # Hint: use quat_error_magnitude
    return 0.0


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    r"""Exponential kernel reward for the global body linear velocity error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \sum_{i=1}^N (v_i - v_{i,ref})^2 \right)

    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = get_body_indices(command, body_names)

    # compute the error
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indices] - command.robot_body_lin_vel_w[:, body_indices]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    r"""Exponential kernel reward for the global body angular velocity error.

    .. math::
        R = \exp\left(-\frac{1}{\sigma^2} \sum_{i=1}^N (w_i - w_{i,ref})^2 \right)

    """
    # extract the used quantities (to enable type-hinting)
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = get_body_indices(command, body_names)

    # compute the error
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indices] - command.robot_body_ang_vel_w[:, body_indices]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    r"""Reward the feet contact time by the amount of time the feet are in contact with the ground.

    .. math::
        R = \sum_{i=1}^N \max(0, t_{last_contact} - t_{threshold}) \cdot \mathbb{1}_{t_{first_air} > 0}

    where :math:`t_{last_contact}` is the time of the last contact, :math:`t_{threshold}` is the threshold
    for the contact time, and :math:`\mathbb{1}_{t_{first_air} > 0}` is the indicator function that is 1 if the
    feet are in the air and 0 otherwise. The reward is zero if the feet are not in the air.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # compute the first air time
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    # compute the last contact time
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    # compute the reward
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
