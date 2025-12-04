#!/usr/bin/env python3
"""
Visualize the STAR robot models (XML and URDF).
This script uses MuJoCo to visualize the robot models.

Usage:
    python visualize_star_robot.py xml    # Visualize MuJoCo XML
    python visualize_star_robot.py urdf   # Visualize URDF (converted to XML)
    python visualize_star_robot.py both   # Compare both side by side
"""

import sys
import mujoco
import mujoco.viewer
import tempfile
import os


def visualize_xml():
    """Visualize the MuJoCo XML model."""
    xml_path = "src/holosoma/holosoma/data/robots/star/star_18dof.xml"
    print(f"Loading XML: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("✓ XML loaded successfully!")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  DOFs: {model.nv}")
    print(f"  Actuators: {model.nu}")
    print("\nLaunching viewer... (Press ESC to exit)")

    mujoco.viewer.launch(model, data)


def visualize_urdf():
    """Visualize the URDF model by converting it to MuJoCo XML format."""
    urdf_path = "src/holosoma/holosoma/data/robots/star/star_18dof.urdf"
    print(f"Loading URDF: {urdf_path}")

    try:
        # Read URDF file
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()

        # Create a temporary directory for conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write URDF to temp location
            temp_urdf = os.path.join(tmpdir, "star_18dof.urdf")
            with open(temp_urdf, 'w') as f:
                f.write(urdf_content)

            # Load URDF using MuJoCo
            model = mujoco.MjModel.from_xml_path(temp_urdf)
            data = mujoco.MjData(model)

            print("✓ URDF loaded successfully!")
            print(f"  Bodies: {model.nbody}")
            print(f"  Joints: {model.njnt}")
            print(f"  DOFs: {model.nv}")
            print(f"  Actuators: {model.nu}")
            print("\nLaunching viewer... (Press ESC to exit)")

            mujoco.viewer.launch(model, data)

    except Exception as e:
        print(f"✗ Error loading URDF: {e}")
        print("\nNote: MuJoCo URDF support may have limitations.")
        print("Consider visualizing the XML version instead.")
        sys.exit(1)


def compare_models():
    """Print comparison of XML and URDF models."""
    xml_path = "src/holosoma/holosoma/data/robots/star/star_18dof.xml"
    urdf_path = "src/holosoma/holosoma/data/robots/star/star_18dof.urdf"

    print("=" * 60)
    print("STAR Robot Model Comparison")
    print("=" * 60)

    # Load XML
    try:
        xml_model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"\n✓ XML Model ({xml_path}):")
        print(f"  Bodies: {xml_model.nbody}")
        print(f"  Joints: {xml_model.njnt}")
        print(f"  DOFs: {xml_model.nv}")
        print(f"  Actuators: {xml_model.nu}")
        print(f"  Sensors: {xml_model.nsensor}")

        # Print body names
        print("\n  Body names:")
        for i in range(xml_model.nbody):
            name = mujoco.mj_id2name(xml_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                print(f"    - {name}")
    except Exception as e:
        print(f"\n✗ Error loading XML: {e}")

    # Load URDF
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_urdf = os.path.join(tmpdir, "star_18dof.urdf")
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            with open(temp_urdf, 'w') as f:
                f.write(urdf_content)

            urdf_model = mujoco.MjModel.from_xml_path(temp_urdf)
            print(f"\n✓ URDF Model ({urdf_path}):")
            print(f"  Bodies: {urdf_model.nbody}")
            print(f"  Joints: {urdf_model.njnt}")
            print(f"  DOFs: {urdf_model.nv}")
            print(f"  Actuators: {urdf_model.nu}")
            print(f"  Sensors: {urdf_model.nsensor}")

            # Print body names
            print("\n  Body names:")
            for i in range(urdf_model.nbody):
                name = mujoco.mj_id2name(urdf_model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    print(f"    - {name}")
    except Exception as e:
        print(f"\n✗ Error loading URDF: {e}")

    print("\n" + "=" * 60)
    print("\nTo visualize, run:")
    print("  python visualize_star_robot.py xml")
    print("  python visualize_star_robot.py urdf")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nDefaulting to XML visualization...")
        visualize_xml()
        return

    mode = sys.argv[1].lower()

    if mode == "xml":
        visualize_xml()
    elif mode == "urdf":
        visualize_urdf()
    elif mode == "both" or mode == "compare":
        compare_models()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
