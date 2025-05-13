# Copyright (c) 2025, Błażej Szargut.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

from bernard import BERNARD_CFG  # isort:skip


##
# Scene definition
##


@configclass
class BernardSceneCfg(InteractiveSceneCfg):
    """A generic configuration for the BERNARD bipedal robot scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

    # robot
    robot: ArticulationCfg = BERNARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=6,
        track_air_time=True,  # filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.1)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=True,
        update_period=0.01,
        # offset=ImuCfg.OffsetCfg(pos=(-0.07007606, -0.07839134, -0.23643290)),
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.5, 1.5),
            ang_vel_z=(-0.1, 0.1),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*_hip.*", ".*_arm.*", ".*_knee.*"], scale=1.0, use_default_offset=True)

    # joint_pos = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*_hip_.*", ".*_arm_.*", ".*_knee_.*"],
    #     scale=9.0,
    #     debug_vis=True,
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        base_orientation = ObsTerm(
            func=mdp.imu_orientation, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        contact_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=".*foot"
                ),
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=0.02,
            clip=(0.0, 1.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.9),
            "dynamic_friction_range": (0.7, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "force_range": (-0.1, 0.1),
    #         "torque_range": (-0.1, 0.1),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
                "z": (-0.7, -0.6),
            },
            "velocity_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
                "roll": (-0.01, 0.01),
                "pitch": (-0.01, 0.01),
                "yaw": (-0.01, 0.01),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # -- helpers
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    alive = RewTerm(
        func=mdp.is_alive,
        weight=2.5,
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.6)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.45}, weight=-20.5)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-3.0e-5)
    dof_torques_limits = RewTerm(func=mdp.applied_torque_limits, weight=-1.0e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-3.5e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*forearm.*|.*arm.*"), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-3.2)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-6.5e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    feet_off_ground = RewTerm(
        func=mdp.both_feet_off_ground,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"),
        }
    )
    dof_pos_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class BernardRoughEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BernardSceneCfg = BernardSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class BernardhRoughEnvCfg_PLAY(BernardRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
