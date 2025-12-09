"""Package containing the motion tracking task for humanoids."""

import os
import toml

# Conveniences to other module directories via relative paths
MOTION_TRACKING_EXT_DIR = os.path.abspath(os.path.join(__file__, "../../"))
"""Path to the extension source directory."""

MOTION_TRACKING_METADATA = toml.load(os.path.join(MOTION_TRACKING_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

MOTION_TRACKING_DATA_DIR = os.path.join(MOTION_TRACKING_EXT_DIR, "data")
"""Path to the extension data directory."""

# Configure the module-level variables
__version__ = MOTION_TRACKING_METADATA["package"]["version"]

##
# Register Gym environments.
##

# import all tasks
from .tasks import *
