import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Motion-Tracking-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motion_tracking_env_cfg:MotionTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionTrackingPPORunnerCfg",
    },
)
