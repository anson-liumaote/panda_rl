# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
Configuration for 12 dof panda.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, RemotizedPDActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from ext_template.assets import ISAACLAB_ASSETS_DATA_DIR

REDDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Panda/reddog_20250221/reddog_20250221/urdf/reddog_20250221/reddog_20250221.usd",
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
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            ".*L_hip_joint": 0.1, 
            ".*R_hip_joint": -0.1,
            "F[LR]_thigh_joint": 0.785,
            "F[LR]_calf_joint": -1.57,
            "R[LR]_thigh_joint": -0.785,
            "R[LR]_calf_joint": 1.57,
        },
        joint_vel={".*": 0.0},
    ),
    # actuators={
    #     "joint_actuator": DCMotorCfg(
    #         joint_names_expr=[".*"],
    #         effort_limit=3.0,
    #         saturation_effort=7.0,
    #         velocity_limit=12.5,
    #         stiffness=8.0,
    #         friction=0.0,
    #         damping=0.1,
    #     ),
    # },
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint_actuator": DelayedPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=3.0,
            # saturation_effort=7.0,
            velocity_limit=12.5,
            stiffness=15.0,
            damping=0.4,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
)

# PANDA_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Panda/PANDA12dof_URDF/PANDA12dof_URDF.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.2),
#         joint_pos={
#             ".*L_hip_joint": 0.1, 
#             ".*R_hip_joint": -0.1,
#             "F[LR]_thigh_joint": 0.785,
#             "F[LR]_calf_joint": -1.57,
#             "R[LR]_thigh_joint": -0.785,
#             "R[LR]_calf_joint": 1.57,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     # actuators={
#     #     "joint_actuator": DelayedPDActuatorCfg(
#     #         joint_names_expr=[".*"],
#     #         effort_limit=2.0,
#     #         stiffness=20.0,
#     #         damping=0.5,
#     #         min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
#     #         max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
#     #     ),
#     # },
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "joint_actuator": DCMotorCfg(
#             joint_names_expr=[".*"],
#             effort_limit=1.5,
#             saturation_effort=1.5,
#             velocity_limit=12.5,
#             stiffness=5.0,
#             damping=0.1,
#             friction=0.0,
#         ),
#     },
# )
