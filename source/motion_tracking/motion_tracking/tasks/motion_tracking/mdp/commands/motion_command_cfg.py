from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .motion_command import MotionCommand


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    motion_file: str = MISSING
    """Path to the motion file to load.

    The motion file should be in the NumPy format. Please check the `scripts/motions`
    folder for the conversion scripts.
    """

    anchor_body_name: str = MISSING
    """Name of the anchor body of the robot to track.

    The anchor body is the name of the body that is used to compute the root pose and velocity.
    """

    body_names: list[str] = MISSING
    """Names of the bodies to track.

    Robots often have multiple bodies that are closely spaced to each other. Tracking all of them
    is not necessary and can be computationally expensive. Therefore, we only track a subset of the
    bodies. The names of the bodies to track are specified by this list.
    """

    pose_range: dict[str, tuple[float, float]] = {}
    """Ranges for resetting the root pose of the robot.

    The pose range is a dictionary with the keys "x", "y", "z", "roll", "pitch", and "yaw".
    The values are the ranges for the corresponding pose components.
    """

    velocity_range: dict[str, tuple[float, float]] = {}
    """Ranges for resetting the root velocity of the robot.

    The velocity range is a dictionary with the keys "x", "y", "z", "roll", "pitch", and "yaw".
    The values are the ranges for the corresponding velocity components.
    """

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Ranges for resetting the joint positions of the robot.

    The joint position range is a tuple of two floats. The first float is the minimum
    value and the second float is the maximum value.
    """

    ##
    # Adaptive sampling settings.
    ##
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
