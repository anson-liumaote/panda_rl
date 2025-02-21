from isaaclab.utils import configclass

from ext_template.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from ext_template.assets.panda import PANDA_CFG

from isaaclab.managers import RewardTermCfg as RewTerm
import ext_template.tasks.locomotion.velocity.mdp as mdp


@configclass
class PandaRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # overwrite links names for panda
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # events
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*f_Link"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*u_Link", "base_link", ".*d_Link", ".*[r,l]_Link"]
        self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2.weight = -2.5e-7

        # self.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)

        # termination
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", ".*u_Link", ".*d_Link", ".*[r,l]_Link"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]



@configclass
class PandaRoughEnvCfg_PLAY(PandaRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
