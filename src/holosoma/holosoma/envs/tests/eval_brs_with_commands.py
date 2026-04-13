"""Evaluate BRS policy with fixed velocity commands and optional video recording.

Usage (headless, no video):
    python eval_brs_with_commands.py --checkpoint <path> --training.headless True

Usage (headless + video saved locally):
    python eval_brs_with_commands.py --checkpoint <path> --logger.headless_recording True

Usage (wandb + video upload):
    python eval_brs_with_commands.py --checkpoint <path> logger:wandb \\
        --logger.headless_recording True
"""

import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG

# ── Velocity commands to apply ─────────────────────────────────────────────
LIN_VEL_X   = 0.5   # m/s forward (negative = backward)
LIN_VEL_Y   = 0.0   # m/s lateral
ANG_VEL_YAW = 0.0   # rad/s turning
# ───────────────────────────────────────────────────────────────────────────


def main():
    init_eval_logging()

    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        config=TYRO_CONIFG,
    )

    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(
        tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval_cmd"
    )
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))

    algo_class = get_class(tyro_config.algo._target_)
    algo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
    algo.load(str(checkpoint))

    # Patch env.step so commands are re-applied after every step.
    # This is needed because eval mode zeros commands on every env reset.
    # The patch is transparent to the algo's video recorder / reward logging.
    _orig_step = env.step

    def _step_with_fixed_commands(action_dict):
        result = _orig_step(action_dict)
        env.simulator.commands[:, 0] = LIN_VEL_X
        env.simulator.commands[:, 1] = LIN_VEL_Y
        env.simulator.commands[:, 2] = ANG_VEL_YAW
        return result

    env.step = _step_with_fixed_commands

    # Also patch reset_all so the very first commands are set correctly.
    _orig_reset_all = env.reset_all

    def _reset_all_with_fixed_commands(*args, **kwargs):
        result = _orig_reset_all(*args, **kwargs)
        env.simulator.commands[:, 0] = LIN_VEL_X
        env.simulator.commands[:, 1] = LIN_VEL_Y
        env.simulator.commands[:, 2] = ANG_VEL_YAW
        return result

    env.reset_all = _reset_all_with_fixed_commands

    print(f"\nRunning with commands: lin_vel_x={LIN_VEL_X}, lin_vel_y={LIN_VEL_Y}, ang_vel_yaw={ANG_VEL_YAW}")
    print(f"Video recording: {tyro_config.logger.headless_recording or tyro_config.logger.video.enabled}")
    print(f"Output dir: {eval_log_dir}\n")

    # Delegate to the standard evaluate_policy — this drives video recording,
    # reward logging, and all callbacks correctly.
    algo.evaluate_policy(max_eval_steps=tyro_config.training.max_eval_steps)

    if simulation_app:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
