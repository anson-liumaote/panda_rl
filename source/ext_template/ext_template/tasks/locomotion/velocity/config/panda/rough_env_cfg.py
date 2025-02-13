from isaaclab.utils import configclass

from ext_template.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from ext_template.assets.panda import PANDA_CFG


@configclass
class PandaRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # overwrite links names for panda
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*d_Link"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*u_Link", "base_link"]
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", ".*u_Link"]



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
