"""Experiment configuration presets for the STAR robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    algo,
    simulator,
    terrain,
)
from holosoma.config_values.loco.star import (
    action,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    termination,
)
from holosoma.config_values import robot

star_18dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="star-18dof-locomotion", name="star_18dof"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.star_18dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.star_18dof_loco_single_wolinvel,
    action=action.star_18dof_joint_pos,
    termination=termination.star_18dof_termination,
    randomization=randomization.star_18dof_randomization,
    command=command.star_18dof_command,
    curriculum=curriculum.star_18dof_curriculum,
    reward=reward.star_18dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

star_18dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="star-18dof-locomotion", name="star_18dof_fast_sac"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.star_18dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.star_18dof_loco_single_wolinvel,
    action=action.star_18dof_joint_pos,
    termination=termination.star_18dof_termination,
    randomization=randomization.star_18dof_randomization,
    command=command.star_18dof_command,
    curriculum=curriculum.star_18dof_curriculum_fast_sac,
    reward=reward.star_18dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

__all__ = ["star_18dof", "star_18dof_fast_sac"]
