#!/usr/bin/env python3
"""
Visualize humanoid_no_hands URDF using PyBullet
Lightweight alternative to Isaac Sim
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import sys


class URDFVisualizer:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        
        # Connect to PyBullet GUI
        self.client = p.connect(p.GUI)
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot URDF
        print(f"Loading URDF: {urdf_path}")
        
        # Starting position and orientation
        start_pos = [0, 0, 1.265]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.robot_id = p.loadURDF(
            urdf_path,
            start_pos,
            start_orientation,
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        print(f"✓ Robot loaded with ID: {self.robot_id}")
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = {}
        self.joint_indices = []
        self.joint_names = []
        
        print(f"\n✓ Robot has {self.num_joints} total joints")
        print("\nJoint Information:")
        print(f"{'Index':<6} {'Name':<30} {'Type':<12} {'Limits'}")
        print("-" * 80)
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]
            
            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit,
                'index': i
            }
            
            # Only track revolute joints
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                
                type_str = "REVOLUTE"
                print(f"{i:<6} {joint_name:<30} {type_str:<12} [{lower_limit:6.2f}, {upper_limit:6.2f}]")
            elif joint_type == p.JOINT_FIXED:
                print(f"{i:<6} {joint_name:<30} {'FIXED':<12}")
        
        print(f"\n✓ Found {len(self.joint_indices)} controllable revolute joints")
        
        # Default standing pose
        self.default_positions = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.3,
            'left_knee_joint': 0.6,
            'left_ankle_roll_joint': 0.0,
            'left_ankle_pitch_joint': -0.3,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.3,
            'right_knee_joint': 0.6,
            'right_ankle_roll_joint': 0.0,
            'right_ankle_pitch_joint': -0.3,
        }
        
        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 1.0]
        )
        
        # Create sliders for joint control
        self.sliders = {}
        self.create_joint_sliders()
        
        # Animation state
        self.animation_mode = 0  # 0: manual, 1: walking, 2: squat
        self.time = 0.0
        
    def create_joint_sliders(self):
        """Create GUI sliders for each joint"""
        print("\n✓ Creating joint control sliders...")
        
        for joint_idx in self.joint_indices:
            joint_name = self.joint_info[joint_idx]['name']
            lower = self.joint_info[joint_idx]['lower']
            upper = self.joint_info[joint_idx]['upper']
            
            # Get default position
            default = self.default_positions.get(joint_name, 0.0)
            
            # Create slider
            slider_id = p.addUserDebugParameter(
                joint_name,
                lower,
                upper,
                default
            )
            
            self.sliders[joint_idx] = slider_id
            
    def reset_to_default(self):
        """Reset robot to default standing pose"""
        print("\n✓ Resetting to default pose...")
        
        for joint_idx in self.joint_indices:
            joint_name = self.joint_info[joint_idx]['name']
            default_pos = self.default_positions.get(joint_name, 0.0)
            
            p.resetJointState(self.robot_id, joint_idx, default_pos)
            
    def animate_walking(self, dt):
        """Simple walking animation"""
        freq = 1.0  # Hz
        
        # Get joint indices by name
        joint_map = {info['name']: idx for idx, info in self.joint_info.items()}
        
        # Hip pitch - alternating
        hip_pitch_amplitude = 0.3
        left_hip_pitch = -0.3 + hip_pitch_amplitude * np.sin(2 * np.pi * freq * self.time)
        right_hip_pitch = -0.3 + hip_pitch_amplitude * np.sin(2 * np.pi * freq * self.time + np.pi)
        
        # Knee - bend more when leg is back
        knee_base = 0.6
        knee_amplitude = 0.4
        left_knee = knee_base + knee_amplitude * (1 + np.sin(2 * np.pi * freq * self.time)) / 2
        right_knee = knee_base + knee_amplitude * (1 + np.sin(2 * np.pi * freq * self.time + np.pi)) / 2
        
        # Ankle pitch - compensate
        left_ankle = -0.3 - 0.2 * np.sin(2 * np.pi * freq * self.time)
        right_ankle = -0.3 - 0.2 * np.sin(2 * np.pi * freq * self.time + np.pi)
        
        # Apply positions
        p.setJointMotorControl2(self.robot_id, joint_map['left_hip_pitch_joint'], 
                               p.POSITION_CONTROL, targetPosition=left_hip_pitch, force=500)
        p.setJointMotorControl2(self.robot_id, joint_map['right_hip_pitch_joint'], 
                               p.POSITION_CONTROL, targetPosition=right_hip_pitch, force=500)
        p.setJointMotorControl2(self.robot_id, joint_map['left_knee_joint'], 
                               p.POSITION_CONTROL, targetPosition=left_knee, force=500)
        p.setJointMotorControl2(self.robot_id, joint_map['right_knee_joint'], 
                               p.POSITION_CONTROL, targetPosition=right_knee, force=500)
        p.setJointMotorControl2(self.robot_id, joint_map['left_ankle_pitch_joint'], 
                               p.POSITION_CONTROL, targetPosition=left_ankle, force=500)
        p.setJointMotorControl2(self.robot_id, joint_map['right_ankle_pitch_joint'], 
                               p.POSITION_CONTROL, targetPosition=right_ankle, force=500)
        
    def animate_squat(self, dt):
        """Simple squat animation"""
        freq = 0.5  # Hz
        
        # Get joint indices by name
        joint_map = {info['name']: idx for idx, info in self.joint_info.items()}
        
        # Symmetric squat
        squat_depth = 0.5
        hip_pitch = -0.3 - squat_depth * (1 + np.sin(2 * np.pi * freq * self.time)) / 2
        knee_angle = 0.6 + squat_depth * 2 * (1 + np.sin(2 * np.pi * freq * self.time)) / 2
        ankle_pitch = -0.3 - squat_depth * 0.5 * (1 + np.sin(2 * np.pi * freq * self.time)) / 2
        
        # Apply to both legs
        for side in ['left', 'right']:
            p.setJointMotorControl2(self.robot_id, joint_map[f'{side}_hip_pitch_joint'], 
                                   p.POSITION_CONTROL, targetPosition=hip_pitch, force=500)
            p.setJointMotorControl2(self.robot_id, joint_map[f'{side}_knee_joint'], 
                                   p.POSITION_CONTROL, targetPosition=knee_angle, force=500)
            p.setJointMotorControl2(self.robot_id, joint_map[f'{side}_ankle_pitch_joint'], 
                                   p.POSITION_CONTROL, targetPosition=ankle_pitch, force=500)
        
    def update_from_sliders(self):
        """Update joint positions from GUI sliders"""
        for joint_idx, slider_id in self.sliders.items():
            target_pos = p.readUserDebugParameter(slider_id)
            
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500
            )
    
    def run(self):
        """Main visualization loop"""
        print("\n" + "="*80)
        print("HUMANOID URDF VISUALIZER - PyBullet")
        print("="*80)
        print("\nControls:")
        print("  - Use sliders on the right panel to control individual joints")
        print("  - Mouse: Left click + drag to rotate camera")
        print("  - Mouse: Scroll to zoom")
        print("  - Press 'R' to reset to default pose")
        print("  - Press 'W' for walking animation")
        print("  - Press 'S' for squat animation")
        print("  - Press 'M' for manual control (stop animation)")
        print("  - Close window to quit")
        print("="*80 + "\n")
        
        # Reset to default pose
        self.reset_to_default()
        
        # Add mode buttons
        mode_text = p.addUserDebugText(
            "Mode: Manual",
            [0, 0, 2.5],
            textColorRGB=[1, 1, 0],
            textSize=1.5
        )
        
        # Main loop
        try:
            while True:
                # Get keyboard events
                keys = p.getKeyboardEvents()
                
                # Check for mode changes
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    self.reset_to_default()
                    self.animation_mode = 0
                    p.removeUserDebugItem(mode_text)
                    mode_text = p.addUserDebugText("Mode: Manual", [0, 0, 2.5], 
                                                   textColorRGB=[1, 1, 0], textSize=1.5)
                    
                elif ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED:
                    self.animation_mode = 1
                    self.time = 0.0
                    p.removeUserDebugItem(mode_text)
                    mode_text = p.addUserDebugText("Mode: Walking", [0, 0, 2.5], 
                                                   textColorRGB=[0, 1, 0], textSize=1.5)
                    
                elif ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
                    self.animation_mode = 2
                    self.time = 0.0
                    p.removeUserDebugItem(mode_text)
                    mode_text = p.addUserDebugText("Mode: Squat", [0, 0, 2.5], 
                                                   textColorRGB=[0, 1, 1], textSize=1.5)
                    
                elif ord('m') in keys and keys[ord('m')] & p.KEY_WAS_TRIGGERED:
                    self.animation_mode = 0
                    p.removeUserDebugItem(mode_text)
                    mode_text = p.addUserDebugText("Mode: Manual", [0, 0, 2.5], 
                                                   textColorRGB=[1, 1, 0], textSize=1.5)
                
                # Update based on mode
                dt = 1.0 / 240.0  # PyBullet default timestep
                
                if self.animation_mode == 1:
                    self.animate_walking(dt)
                    self.time += dt
                elif self.animation_mode == 2:
                    self.animate_squat(dt)
                    self.time += dt
                else:
                    # Manual control from sliders
                    self.update_from_sliders()
                
                # Step simulation
                p.stepSimulation()
                time.sleep(dt)
                
        except KeyboardInterrupt:
            print("\n✓ Visualization stopped by user")
        
        finally:
            p.disconnect()
            print("✓ PyBullet disconnected")


def main():
    # Get URDF path from command line or use default
    if len(sys.argv) > 1:
        urdf_path = sys.argv[1]
    else:
        urdf_path = "humanoid_no_hands.urdf"
    
    print(f"Starting URDF Visualizer...")
    print(f"URDF file: {urdf_path}\n")
    
    try:
        visualizer = URDFVisualizer(urdf_path)
        visualizer.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())