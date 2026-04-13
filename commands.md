Train (IsaacSim, full terrain mix):


conda run -n lab python src/holosoma/train.py exp:brs-v1
Train (MuJoCo, flat plane — faster iteration, matches viewer):


conda run -n lab python src/holosoma/train.py exp:brs-v1-mujoco
Resume training from checkpoint:


conda run -n lab python src/holosoma/train.py exp:brs-v1 \
    --training.resume logs/brs-v1-locomotion/20260411_124609-brs_v1-locomotion/model_24999.pt
View policy in MuJoCo:


DISPLAY=:14 conda run -n gym_env python src/holosoma/holosoma/envs/tests/view_brs_mujoco.py \
    --checkpoint logs/brs-v1-locomotion/20260411_124609-brs_v1-locomotion/model_24999.pt \
    --lin_vel_x 0.5 --duration 30
TensorBoard:


conda run -n lab tensorboard --logdir logs/brs-v1-locomotion/