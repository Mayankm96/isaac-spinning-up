# BeyondMimic Motion Tracking Tutorial

## Overview

This tutorial provides a minimal, well-structured Isaac Lab extension template and walks you through how to integrate a complex control framework — **BeyondMimic** — as an example.
The goal is to help you learn how to build, package, and run custom Isaac Lab extensions while understanding how real research-grade controllers can be integrated into the ecosystem.

You can use this template as a starting point for:

* RL training pipelines
* Motion imitation & motion tracking
* Robotics benchmarks
* Controller prototyping

**Keywords:** motion-tracking, deepmimic, humanoids

https://github.com/user-attachments/assets/838a7ef1-795d-4365-b1c3-9942083de6d3

## 🤖 About BeyondMimic

[BeyondMimic](https://arxiv.org/abs/2508.08241) is a versatile humanoid control framework that provides:

* Highly dynamic motion tracking using diffusion-based policies
* State-of-the-art motion quality, suitable for real-world deployment
* Steerable, test-time control with guided diffusion-based controllers

This tutorial includes the training pipeline for the whole-body tracking controller.
The implementation is based on the original implementation by the authors on
[GitHub](https://github.com/HybridRobotics/whole_body_tracking).

## 📁 Repository Structure

You will be working mainly inside:

```bash
source/
  motion_tracking/
    data/
      motions/  # location of all csv, pkl, npy motion trajectories
      unitree_g1/ # the URDF/USD files of the asset
    motion_tracking/
      assets/
      dataset/
      tasks/
        motion_tracking/ # <--- !!! Core Exercise File !!!
      __init__.py
scripts/
  motions/  # utility scripts to convert motion data formats
  rsl_rl/  # training and playing scripts based on RSL-RL
README.md
```

## ⚙️ Installation

* Install Isaac Lab v2.3.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

* Clone this repository separately from the Isaac Lab installation
  (i.e., outside the `IsaacLab` directory):

  ```bash
  # Option 1: SSH
  git clone git@github.com:mayankm96/isaac-spinning-up.git

  # Option 2: HTTPS
  git clone https://github.com/mayankm96/isaac-spinning-up.git
  ```

* Using a Python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/motion_tracking
  ```

## 🦾 Motion Tracking

Motion tracking is the core component of BeyondMimic-style humanoid control.
It measures how closely the simulated agent follows a reference motion, and provides rewards that guide the agent during training.

In this project, motion tracking consists of:

* Global Root Tracking – position and orientation of the humanoid’s base or anchor.
* Relative Body Tracking – positions and orientations of key limbs relative to the root.
* Exponential Kernel Rewards – smooth rewards that penalize deviation without clipping.
* Reference Motion Switching – allows multiple motion clips to be used during training for robustness.

## Motion Preprocessing & Registry Setup

The reference motion should be retargeted and use generalized coordinates only.

For the time being, we use the pre-processed files pre-included in this tutorial.

<details close>
<summary>Instructions</summary>

Gather the reference motion datasets (please follow the original licenses),
we use the same convention as .csv of Unitree's dataset

* Unitree-retargeted LAFAN1 Dataset is available
  on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
* Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
* Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
* Balance motions are from [HuB](https://hub-robot.github.io/)

The conversion script adds smooth transitions from a safe standing pose at the
start and end of the motion to ensure safe deployment. Since the motion is
cyclic, we also repeat it to reach the desired duration.

As an example, we use the spin kick data using the link referenced in the MimicKit
[installation instructions](https://github.com/xbpeng/MimicKit?tab=readme-ov-file#installation).

```bash
python scripts/motions/pkl_to_csv.py \
    --pkl-file source/motion_tracking/data/motions/pkl/g1_spinkick.pkl \
    --csv-file source/motion_tracking/data/motions/csv/g1_spinkick.csv \
    --duration 2.65 \
    --add-start-transition \
    --add-end-transition \
    --transition-duration 0.5 \
    --pad-duration 1.0
```

```bash
python scripts/motions/csv_to_npz.py \
    --input_file source/motion_tracking/data/motions/csv/g1_spinkick.csv \
    --input_fps 30 \
    --output_file source/motion_tracking/data/motions/npz/g1_spinkick.npz \
    --output_fps 50 \
    --headless
```

There are additional trajectories available inside the [`source/motion_tracking/data/motions/pkl`](motion_tracking/data/motions/pkl) directory to try out!

</details>

### Dummy Agents

Check/debug the simulation environment by the following command:

```bash
python scripts/zero_agent.py --task=Motion-Tracking-G1-v0 --num_envs 32
```

### Policy Training

Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Motion-Tracking-G1-v0 --headless
```

After running the above, the training logs are saved into [`logs`](logs) directory.
To view the training curves, open the tensorboard logger:

```bash
cd /PATH/TO/REPO
tensorboard --logdir .
```

### Policy Evaluation

Play the trained policy by the following command. This automatically loads
the last checkpoint saved for the environment.

```bash
python scripts/rsl_rl/play.py --task=Motion-Tracking-G1-v0 --num_envs=50
```

You can compare this policy to the pre-trained policy included in the repository
at the location [`scripts/rsl_rl/checkpoints`](scripts/rsl_rl/checkpoints)

```bash
python scripts/rsl_rl/play.py --task=Motion-Tracking-G1-v0 --num_envs=50 --use_pretrained_checkpoint
```

To record a video, you can run the `play.py` script with additional commands:

```bash
python scripts/rsl_rl/play.py --task=Motion-Tracking-G1-v0 --num_envs=16 --video --video_length 500 --headless --enable_cameras
```

## 🧪 Exercise Overview

In this tutorial, you will progressively build a robust humanoid motion-tracking pipeline
inside an Isaac Lab extension. Each step corresponds to modifying or implementing specific
functions inside your task folder:

```bash
source/motion_tracking/motion_tracking/tasks/motion_tracking
```

Specifically, you will write the terms that define the Markov Decision Process for motion
tracking task.

Each section contains TODO blocks for you to fill in.

By completing the motion tracking exercises, you will learn to:

* Compute position and orientation errors for humanoid bodies
* Implement exponential-kernel rewards for imitation learning
* Pre-process motion data for RL training
* Integrate multiple reference motions into an environment
* Enhance sim-to-sim and sim-to-real robustness

### Step 1: Define tracking reward function

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/mdp/rewards.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/mdp/rewards.py)

Implement the reward terms responsible for tracking the reference motion.

### Step 2: Define termination function

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/mdp/terminations.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/mdp/terminations.py)

Implement the termination terms responsible for terminating when bad tracking happens.

### Step 3: Tune the learning agent

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/agents/rsl_rl_ppo_cfg.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/agents/rsl_rl_ppo_cfg.py)

Enable empirical normalization to the actor and critic and retrain the policy. Notice the learning curves.

### Step 4: Define domain randomization for friction

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py)

In this step, you will implement surface friction randomization to make the humanoid policy robust to different contact conditions.
This models uncertainty in real-world contact surfaces, such as slippery floors, grippy mats, or variations in shoe soles.

Isaac Lab allows you to randomize contact parameters (like friction, restitution, damping, etc.) through the `physics_material` field inside your task’s `EventsCfg`.
This configuration controls how materials are updated during environment resets, without manually editing rigid-body properties.

Modify these setting to make policy robust to friction. You can try setting these to small values
to check the current trained policy.

### Step 5: Define more domain randomization for robustness

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py)

In Isaac Lab, domain randomization is not limited to friction or material properties.
The EventsCfg system allows you to register event-driven perturbations that run at different moments:

* startup — runs once when the simulation is created
* reset — runs at each episode reset
* interval — runs periodically during rollout (good for random pushes)

There are several example EventTerm definitions that have been commented out.
Your task is to inspect them, understand what they do, and optionally enable or rewrite them as part of the robustness exercises.

### Step 6: Remove the IMU From the Observation Space

File: [`source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py`](source/motion_tracking/motion_tracking/tasks/motion_tracking/motion_tracking_env_cfg.py)

In many real-world humanoid robots, onboard IMUs can fail, drift, or be absent depending on the hardware configuration.
A robust controller should be able to maintain balance and track motion even without direct inertial measurements.

In this exercise, you will remove all IMU-related signals from the observation vector and modify the MDP accordingly.

### Step 7: Switch motions to other references

The folder [`source/motion_tracking/data/motions`](source/motion_tracking/data/motions) contains additional motions to try out.

Your task in this step is to pre-process, load, and plug in new reference motions into your MDP.
Re-train the policies for these new references and tune the parameters to improve the tracking.

## 🎯 Optional Challenge Questions

* How does the choice of std influence the "sharpness" of each reward?
* Why do exponential-kernel rewards often stabilize training better than linear penalties?
* Which terms do you expect to be most important early in training? Why?

## 🧹 Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Acknowledgements

This project is provided primarily for educational and research purposes.

We gratefully acknowledge the following open-source projects, which inspired and/or contributed code to this repository:

* [`HybridRobotics/whole_body_tracking`](https://github.com/HybridRobotics/whole_body_tracking)
* [`mujocolab/g1_spinkick_example`](https://github.com/mujocolab/g1_spinkick_example)

All third-party code is used in accordance with its original license.  
Copies of the relevant licenses can be found in the [`docs/licenses`](docs/licenses/) directory.

Some files have been modified from their original versions.
