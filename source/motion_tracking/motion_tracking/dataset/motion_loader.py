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

import numpy as np
import os
import torch
from typing import Sequence


class MotionLoader:
    """Helper class to load and sample motion data from NumPy-file format.

    This class is used to load the motion data from a NumPy file and provide the motion data
    in the world frame. The motion data is stored in the following format:

    - fps: Frames per second.
    - joint_pos: Joint positions. Shape: (time_steps, num_joints).
    - joint_vel: Joint velocities. Shape: (time_steps, num_joints).
    - body_pos_w: Body positions in the world frame. Shape: (time_steps, num_bodies, 3).
    - body_quat_w: Body quaternions in the world frame. Shape: (time_steps, num_bodies, 4).
    - body_lin_vel_w: Body linear velocities in the world frame. Shape: (time_steps, num_bodies, 3).
    - body_ang_vel_w: Body angular velocities in the world frame. Shape: (time_steps, num_bodies, 3).

    The data is processed from the CSV (BeyondMimic data) or PKL (MimicKit data) to the NumPy file format.
    Please check the `scripts/motions` folder for the conversion scripts.
    """

    def __init__(self, motion_file: str, body_indices: Sequence[int], device: str = "cpu"):
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            body_indices: Indexes of the bodies to load.
            device: The device to which to load the data. Defaults to "cpu".
        """
        # check if the motion file exists
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        # initialize the internal variables
        self._motion_file = motion_file
        self._body_indices = body_indices
        # load the motion file
        data = np.load(self._motion_file)
        self.fps = data["fps"]
        self._joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self._joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        # additional properties
        self.time_step_total = self.joint_pos.shape[0]

    def __str__(self):
        msgs = []
        # print the motion file information
        msgs.append(f"Motion file loaded: {self._motion_file}")
        msgs.append(f"  |-- FPS: {self.fps}")
        msgs.append(f"  |-- Time steps: {self.time_step_total}")
        msgs.append(f"  |-- Body indices: {self._body_indices}")
        return "\n".join(msgs)

    """
    Properties.
    """

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Body positions in the world frame. Shape: (time_steps, num_bodies, 3)."""
        return self._body_pos_w[:, self._body_indices]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Body quaternions in the world frame. Shape: (time_steps, num_bodies, 4).

        The quaternion is in the format of (w, x, y, z).
        """
        return self._body_quat_w[:, self._body_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Body linear velocities in the world frame. Shape: (time_steps, num_bodies, 3)."""
        return self._body_lin_vel_w[:, self._body_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Body angular velocities in the world frame. Shape: (time_steps, num_bodies, 3)."""
        return self._body_ang_vel_w[:, self._body_indices]

    @property
    def joint_pos(self) -> torch.Tensor:
        """Joint positions. Shape: (time_steps, num_joints)."""
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        """Joint velocities. Shape: (time_steps, num_joints)."""
        return self._joint_vel
