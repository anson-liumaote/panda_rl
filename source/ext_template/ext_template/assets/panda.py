# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_CFG`: The Spot robot with delay PD and remote PD actuators.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ext_template.assets import ISAACLAB_ASSETS_DATA_DIR

PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Panda/reddog/urdf/reddog/reddog.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,  # all knees
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=10.0,
            stiffness=20.0,
            damping=0.5,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
)
