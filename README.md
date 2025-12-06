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
        motion_tracking/ # <--- Core Exercise File
      __init__.py
scripts/
  motions/  # utility scripts to convert motion data formats
  rsl_rl/  # training and playing scripts based on RSL-RL
README.md
```

## Installation

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

## Motion Tracking

### Motion Preprocessing & Registry Setup

The reference motion should be retargeted and use generalized coordinates only.

<details open>
<summary>Instructions</summary>

* Gather the reference motion datasets (please follow the original licenses),
  we use the same convention as .csv of Unitree's dataset

  * Unitree-retargeted LAFAN1 Dataset is available
    on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
  * Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
  * Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
  * Balance motions are from [HuB](https://hub-robot.github.io/)

* Convert retargeted motions to include the maximum coordinates information
  (body pose, body velocity, and body acceleration) via forward kinematics:

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

</details>

### Policy Training

Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Motion-Tracking-G1-v0 \
    --headless \
    --logger wandb \
    --log_project_name unitree_g1_motion \
    --run_name sidekick
```

### Policy Evaluation

Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Motion-Tracking-G1-v0 --num_envs=50
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
