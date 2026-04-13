"""Experiment configuration presets for the BRS robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    algo,
    simulator,
    terrain,
)
from holosoma.config_values.loco.brs import (
    action,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    termination,
)
from holosoma.config_values import robot

brs_v1 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="brs-v1-locomotion", name="brs_v1"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacsim,
    robot=robot.brs_v1,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.brs_v1_loco_single_wolinvel,
    action=action.brs_v1_joint_pos,
    termination=termination.brs_v1_termination,
    randomization=randomization.brs_v1_randomization,
    command=command.brs_v1_command,
    curriculum=curriculum.brs_v1_curriculum,
    reward=reward.brs_v1_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

brs_v1_mujoco = replace(
    brs_v1,
    training=TrainingConfig(project="brs-v1-locomotion", name="brs_v1_mujoco"),
    simulator=simulator.mujoco,
    terrain=terrain.terrain_locomotion_plane,
)

__all__ = ["brs_v1", "brs_v1_mujoco"]
