"""Standalone MuJoCo viewer for BRS policy.

Obs vector is assembled in ALPHABETICAL order to match training
(holosoma/managers/observation/manager.py sorts keys before cat).

Alphabetical order of actor_obs terms:
    actions(12) | base_ang_vel(3)*0.25 | command_ang_vel(1) |
    command_lin_vel(2) | cos_phase(2) | dof_pos(12) |
    dof_vel(12)*0.05 | projected_gravity(3) | sin_phase(2)
    = 49 dims total

Torque formula from joint_control.py:
    torque = kp * (action * action_scale + default_pos - dof_pos) - kd * dof_vel

Usage:
    DISPLAY=:14 conda run -n gym_env python src/holosoma/holosoma/envs/tests/view_brs_mujoco.py \\
        --checkpoint logs/brs-v1-locomotion/20260411_124609-brs_v1-locomotion/model_24999.pt \\
        --lin_vel_x 0.5 --duration 30
"""

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

# ── Commands ──────────────────────────────────────────────────────────────
LIN_VEL_X   = 0.5
LIN_VEL_Y   = 0.0
ANG_VEL_YAW = 0.0
# ─────────────────────────────────────────────────────────────────────────

XML_PATH = "src/holosoma/holosoma/data/robots/brs/brs_v1.xml"

# DOF order must match training (robot.py dof_names, left leg first)
TRAIN_DOF_ORDER = [
    "left_hip_yaw",   "left_hip_roll",   "left_hip_pitch",  "left_knee_pitch",
    "left_ankle_roll", "left_ankle_pitch",
    "right_hip_yaw",  "right_hip_roll",  "right_hip_pitch", "right_knee_pitch",
    "right_ankle_roll","right_ankle_pitch",
]

DEFAULT_JOINT_ANGLES = {
    "left_hip_yaw": 0.0,   "left_hip_roll": 0.0,   "left_hip_pitch": -0.3,
    "left_knee_pitch": 0.6, "left_ankle_roll": 0.0,  "left_ankle_pitch": -0.3,
    "right_hip_yaw": 0.0,  "right_hip_roll": 0.0,  "right_hip_pitch": -0.3,
    "right_knee_pitch": 0.6,"right_ankle_roll": 0.0, "right_ankle_pitch": -0.3,
}

# Exact values from robot.py stiffness/damping dicts
KP = {"hip_yaw": 40.0, "hip_roll": 100.0, "hip_pitch": 40.0,
      "knee_pitch": 100.0, "ankle_roll": 30.0, "ankle_pitch": 30.0}
KD = {"hip_yaw": 2.5,  "hip_roll": 6.0,   "hip_pitch": 2.5,
      "knee_pitch": 6.0, "ankle_roll": 2.0,  "ankle_pitch": 2.0}

ACTION_SCALE  = 0.25   # robot.py control.action_scale
EFFORT_LIMIT  = 120.0  # Nm, clamp from robot.py
SIM_DT        = 0.002  # 500 Hz physics
CTRL_DEC      = 10     # control every 10 steps → 50 Hz
GAIT_PERIOD   = 1.0    # seconds


def gains_for(dof_name):
    for key in KP:
        if key in dof_name:
            return KP[key], KD[key]
    raise ValueError(f"No gains for '{dof_name}'")


def build_actor(state_dict):
    prefix = "actor_module.module"
    layers = []
    for i in [0, 2, 4, 6]:
        w = state_dict[f"{prefix}.{i}.weight"]
        b = state_dict[f"{prefix}.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = w
        lin.bias.data = b
        layers.append(lin)
        if i < 6:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def quat_to_rot(q_wxyz):
    """MuJoCo [w,x,y,z] → 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--lin_vel_x",   type=float, default=LIN_VEL_X)
    parser.add_argument("--lin_vel_y",   type=float, default=LIN_VEL_Y)
    parser.add_argument("--ang_vel_yaw", type=float, default=ANG_VEL_YAW)
    parser.add_argument("--duration",    type=float, default=30.0)
    args = parser.parse_args()
    command = np.array([args.lin_vel_x, args.lin_vel_y, args.ang_vel_yaw], dtype=np.float32)

    # ── MuJoCo setup ───────────────────────────────────────────────────
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)
    mj_model.opt.timestep = SIM_DT

    def jid(name):
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_", "-"))
        if i < 0:
            raise ValueError(f"Joint '{name}' not found")
        return i

    train_qposadr = np.array([mj_model.jnt_qposadr[jid(n)] for n in TRAIN_DOF_ORDER])
    train_dofadr  = np.array([mj_model.jnt_dofadr[jid(n)]  for n in TRAIN_DOF_ORDER])
    default_pos   = np.array([DEFAULT_JOINT_ANGLES[n] for n in TRAIN_DOF_ORDER], dtype=np.float32)
    kp_arr = np.array([gains_for(n)[0] for n in TRAIN_DOF_ORDER], dtype=np.float32)
    kd_arr = np.array([gains_for(n)[1] for n in TRAIN_DOF_ORDER], dtype=np.float32)

    act_name_to_ctrl = {mj_model.actuator(i).name: i for i in range(mj_model.nu)}
    train_to_ctrl = np.array([act_name_to_ctrl[n.replace("_", "-")] for n in TRAIN_DOF_ORDER])

    # ── Reset ──────────────────────────────────────────────────────────
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2] = 1.1136   # foot bottom just touches floor
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(TRAIN_DOF_ORDER):
        mj_data.qpos[train_qposadr[i]] = default_pos[i]
    mujoco.mj_forward(mj_model, mj_data)

    # ── Policy ─────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(ckpt["actor_model_state_dict"])
    actor.eval()

    print(f"Commands: vx={command[0]:.2f}  vy={command[1]:.2f}  wz={command[2]:.2f}")
    print("Opening MuJoCo viewer...")

    # Previous action (starts at zero)
    prev_action = np.zeros(12, dtype=np.float32)
    target_pos  = default_pos.copy()

    # Gait phase: matches training eval init — phase_offset = [0, -π]
    # Training update: phase = fmod(step * phase_dt + phase_offset + π, 2π) - π
    # phase_dt = 2π * ctrl_dt * gait_freq = 2π * (1/50) * 1.0
    phase_offset = np.array([0.0, -math.pi], dtype=np.float32)
    phase_dt_per_step = 2.0 * math.pi / (50.0 * GAIT_PERIOD)  # per policy step
    ctrl_step = 0  # counts policy (50 Hz) steps

    step = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance  = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        t_start = time.time()
        while viewer.is_running() and (time.time() - t_start) < args.duration:
            step_start = time.time()

            # ── Policy inference at 50 Hz ──────────────────────────────
            if step % CTRL_DEC == 0:
                # Update gait phase — matches training LocomotionGait.step():
                #   phase = fmod(episode_step * phase_dt + phase_offset + π, 2π) - π
                phase_raw = ctrl_step * phase_dt_per_step + phase_offset
                phase = np.fmod(phase_raw + math.pi, 2.0 * math.pi) - math.pi
                ctrl_step += 1

                q  = mj_data.qpos[3:7]          # [w,x,y,z]
                R  = quat_to_rot(q)
                proj_grav = (R.T @ np.array([0., 0., -1.])).astype(np.float32)

                dof_pos = (mj_data.qpos[train_qposadr] - default_pos).astype(np.float32)
                dof_vel = (mj_data.qvel[train_dofadr] * 0.05).astype(np.float32)
                ang_vel = (mj_data.qvel[3:6] * 0.25).astype(np.float32)

                # ── ALPHABETICAL obs order (matches training manager.py) ──
                # actions(12) | base_ang_vel(3) | command_ang_vel(1) |
                # command_lin_vel(2) | cos_phase(2) | dof_pos(12) |
                # dof_vel(12) | projected_gravity(3) | sin_phase(2)
                cos_ph = np.cos(phase).astype(np.float32)  # [cos_left, cos_right]
                sin_ph = np.sin(phase).astype(np.float32)  # [sin_left, sin_right]

                obs = np.concatenate([
                    prev_action,            # 12  actions
                    ang_vel,                # 3   base_ang_vel * 0.25
                    command[2:3],           # 1   command_ang_vel
                    command[:2],            # 2   command_lin_vel
                    cos_ph,                 # 2   cos_phase
                    dof_pos,                # 12  dof_pos
                    dof_vel,                # 12  dof_vel * 0.05
                    proj_grav,              # 3   projected_gravity
                    sin_ph,                 # 2   sin_phase
                ], dtype=np.float32)        # = 49 total

                with torch.no_grad():
                    action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

                prev_action = action.copy()
                target_pos  = default_pos + action * ACTION_SCALE

            # ── Torque = kp*(target - pos) - kd*vel  (exact framework formula) ──
            dof_pos_raw = mj_data.qpos[train_qposadr]
            dof_vel_raw = mj_data.qvel[train_dofadr]
            torques = kp_arr * (target_pos - dof_pos_raw) - kd_arr * dof_vel_raw
            torques = np.clip(torques, -EFFORT_LIMIT, EFFORT_LIMIT)

            mj_data.ctrl[train_to_ctrl] = torques
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            step += 1

            if step % 200 == 0:
                x, y, h = mj_data.qpos[0], mj_data.qpos[1], mj_data.qpos[2]
                print(f"t={mj_data.time:.1f}s  x={x:.3f}  y={y:.3f}  h={h:.3f}m  ncon={mj_data.ncon}")

            # Real-time pacing
            slack = SIM_DT - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)

    print(f"Done. Simulated {mj_data.time:.1f}s")


if __name__ == "__main__":
    main()
