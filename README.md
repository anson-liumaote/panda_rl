# README

# Motion Imitation Using Reinforcement Learning and Animation Data

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repo extracts motion data from Unity animations and trains reinforcement learning models in Isaac Lab, enabling a real robot to imitate the animated movements.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.
- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPSgit clone https://github.com/isaac-sim/IsaacLabExtensionTemplate.git
# Option 2: SSHgit clone git@github.com:isaac-sim/IsaacLabExtensionTemplate.git
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/ext_template
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
```

## Run

- Extract joint positions, velocities, and foot endpoints from pre-recorded animation data

```bash
python scripts/motion_converter/50hz_sampling.py
python scripts/motion_converter/endpoint-recorder.py
```

- Move and save joint positions, velocities, and foot endpoints data in source/ext_template/data/Animation
- Edit the observation function’s `max_count` to match the motion data’s step count.

```python
current_step = ObsTerm(func=spot_mdp.motion_sequence_counter, params={"asset_cfg": SceneEntityCfg("robot"), "max_count":83})
```

- Edit the `file_path` in reward functions to match the motion data’s files
- Run training command

```bash
python scripts/rsl_rl/train.py --task=Template-Isaac-Velocity-Flat-Panda-v2 --headless --video
```

## Troubleshooting
