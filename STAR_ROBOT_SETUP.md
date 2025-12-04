# STAR 18-DOF Robot Configuration for Holosoma

This document describes the complete setup for training the STAR 18-DOF humanoid robot on rough terrains using the Holosoma framework.

## Robot Specifications

- **Name**: STAR (based on star_18dof.xml)
- **DOF**: 12 actuated joints (6 per leg, no arms/waist)
- **Joint Layout**:
  - Each leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
  - Symmetric left/right configuration

### Joint Limits (from XML)

| Joint | Range (degrees) | Torque Limit (N·m) |
|-------|-----------------|-------------------|
| Hip Yaw | -50° to 20° | 351 |
| Hip Roll | -20° to 8° | 351 |
| Hip Pitch | -30° to 40° | 298 |
| Knee | -5° to 100° | 420 |
| Ankle Pitch | -30° to 30° | 262 |
| Ankle Roll | -40° to 25° | 131 |

## Files Created

### 1. Robot Model
- `src/holosoma/holosoma/data/robots/star/star_18dof.xml` - MuJoCo XML model

### 2. Robot Configuration
- `src/holosoma/holosoma/config_values/robot.py` - Added `star_18dof` configuration

### 3. Locomotion Configurations
```
src/holosoma/holosoma/config_values/loco/star/
├── __init__.py
├── action.py           # Joint position control
├── command.py          # Velocity commands and gait
├── curriculum.py       # Progressive difficulty
├── experiment.py       # Main experiment configs
├── observation.py      # State observations
├── randomization.py    # Domain randomization
├── reward.py           # Reward functions
└── termination.py      # Episode termination conditions
```

## Training Commands

### Setup Environment

First, set up IsaacGym (recommended for fast training):
```bash
cd /media/sarvesh/nithin/holosoma
bash scripts/setup_isaacgym.sh
source scripts/source_isaacgym_setup.sh
```

### Training on Rough Terrain

#### Option 1: PPO Algorithm
```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:star-18dof \
    simulator:isaacgym \
    terrain:terrain-locomotion-mix \
    logger:wandb \
    --training.seed 1
```

#### Option 2: FastSAC Algorithm (Faster Convergence)
```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:star-18dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-mix \
    logger:wandb \
    --training.seed 1
```

### Customizing Terrain Difficulty

Increase rough terrain percentage and difficulty:
```bash
python src/holosoma/holosoma/train_agent.py \
    exp:star-18dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-mix \
    --terrain.terrain-term.terrain-config='{"flat": 0.1, "rough": 0.7, "low_obstacles": 0.2}' \
    --terrain.terrain-term.amplitude-range=[0.02,0.08] \
    --terrain.terrain-term.num-rows=15 \
    logger:wandb
```

### Multi-GPU Training

For faster training with multiple GPUs:
```bash
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:star-18dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-mix \
    --training.num-envs 16384 \
    logger:wandb
```

### Start with Flat Terrain (Recommended)

For initial testing, start with flat terrain to verify the configuration:
```bash
python src/holosoma/holosoma/train_agent.py \
    exp:star-18dof \
    simulator:isaacgym \
    terrain:terrain-locomotion-plane \
    logger:wandb \
    --training.seed 1
```

Once the robot can walk on flat terrain, switch to mixed terrain.

## Evaluation

After training, evaluate your policy:

```bash
# Evaluate from Wandb checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=wandb://<ENTITY>/<PROJECT>/<RUN_ID>/<CHECKPOINT_NAME>

# Evaluate from local checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=/path/to/checkpoint.pt
```

### Interactive Control During Evaluation

When the simulator window is active, use keyboard controls:
- `w`/`s` - Forward/backward velocity
- `a`/`d` - Left/right strafe
- `q`/`e` - Rotate left/right
- `z` - Stop (zero velocity)

## Configuration Details

### Terrain Configuration

The default mixed terrain (`terrain-locomotion-mix`) includes:
- 20% flat terrain
- 60% rough terrain (height variations)
- 20% low obstacles

Terrain difficulty increases across rows (0-9), with row 0 being easiest.

### Training Parameters

#### PPO (star-18dof)
- Learning iterations: 25,000
- Environments: 4,096
- Uses symmetry
- Curriculum learning enabled

#### FastSAC (star-18dof-fast-sac)
- Learning iterations: 50,000
- Environments: 4,096
- Uses symmetry
- More aggressive curriculum

### Reward Function

The robot is rewarded for:
- Tracking linear velocity commands (weight: 2.0)
- Tracking angular velocity commands (weight: 1.5)
- Proper gait with foot swing height (weight: 5.0)
- Staying alive (weight: 1.0)

Penalties for:
- Body orientation deviation
- Unwanted angular velocities
- Rapid action changes
- Incorrect joint poses
- Feet too close together
- Incorrect foot orientation

### Domain Randomization

Training includes randomization of:
- Mass properties (links and base)
- Friction coefficients
- Center of mass offsets
- PD gains
- External pushes
- Action delays

This helps the policy generalize to real hardware.

## Troubleshooting

### If training fails to start:
1. Verify IsaacGym setup: `source scripts/source_isaacgym_setup.sh`
2. Check GPU availability: `nvidia-smi`
3. Reduce number of environments: `--training.num-envs 2048`

### If robot falls immediately:
1. Start with flat terrain first
2. Check joint limits in XML match your hardware
3. Adjust PD gains in `robot.py` if needed

### If training is slow:
1. Disable video logging: `--logger.video.enabled=False`
2. Use FastSAC instead of PPO
3. Use multiple GPUs with torchrun

## Next Steps

1. **Test on flat terrain first** to verify configuration
2. **Monitor training** via Wandb dashboard
3. **Increase terrain difficulty** gradually
4. **Export to ONNX** for deployment: `--training.export_onnx=True`
5. **Deploy to real robot** using holosoma_inference

## Notes

- All joint limits and torque limits from your original `star_18dof.xml` have been preserved
- The robot configuration uses appropriate PD gains for the joint torque limits
- The framework automatically handles symmetry between left/right legs
- Training typically takes 2-4 hours on a modern GPU (RTX 3090 or better)

## References

- Main README: `/media/sarvesh/nithin/holosoma/README.md`
- Training Guide: `/media/sarvesh/nithin/holosoma/src/holosoma/README.md`
- Your robot XML: `/media/sarvesh/nithin/holosoma/star_18dof.xml` (original)
- Framework XML: `/media/sarvesh/nithin/holosoma/src/holosoma/holosoma/data/robots/star/star_18dof.xml`
