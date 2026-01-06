"""
Visualize Evolved Creatures - CM3020 AI Coursework Part B

This script loads a saved creature from CSV and visualizes it climbing
the mountain in a PyBullet GUI window.

Usage:
    python visualize_creature.py creature_file.csv
    python visualize_creature.py --random  # Generate and view a random creature

Author: Student
Date: January 2026
"""

import argparse
import sys
import time
import pybullet as p
import pybullet_data
import creature
import genome
import os


def setup_mountain_environment(arena_size=20):
    """Set up the mountain climbing environment for visualization."""
    
    # Connect to GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Set gravity
    p.setGravity(0, 0, -10)
    
    # Create arena floor
    wall_thickness = 0.5
    floor_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, wall_thickness]
    )
    floor_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, arena_size/2, wall_thickness],
        rgbaColor=[1, 1, 0, 1]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=floor_collision,
        baseVisualShapeIndex=floor_visual,
        basePosition=[0, 0, -wall_thickness]
    )
    
    # Create arena walls
    wall_height = 1
    wall_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2]
    )
    wall_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[arena_size/2, wall_thickness/2, wall_height/2],
        rgbaColor=[0.7, 0.7, 0.7, 1]
    )
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                     baseVisualShapeIndex=wall_visual,
                     basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                     baseVisualShapeIndex=wall_visual,
                     basePosition=[0, -arena_size/2, wall_height/2])
    
    wall_collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2]
    )
    wall_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[wall_thickness/2, arena_size/2, wall_height/2],
        rgbaColor=[0.7, 0.7, 0.7, 1]
    )
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                     baseVisualShapeIndex=wall_visual,
                     basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                     baseVisualShapeIndex=wall_visual,
                     basePosition=[-arena_size/2, 0, wall_height/2])
    
    # Load the mountain
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shapes_dir = os.path.join(script_dir, 'shapes')
    mountain_position = (0, 0, -1)
    mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
    p.setAdditionalSearchPath(shapes_dir)
    mountain = p.loadURDF(
        "gaussian_pyramid.urdf",
        mountain_position,
        mountain_orientation,
        useFixedBase=1
    )
    
    return mountain


def load_creature_from_csv(csv_file):
    """Load a creature from a saved CSV file."""
    dna = genome.Genome.from_csv(csv_file)
    cr = creature.Creature(gene_count=1)
    cr.update_dna(dna)
    return cr


def visualize_creature(cr, duration=20):
    """
    Visualize a creature in the mountain environment.
    
    Args:
        cr: Creature object to visualize
        duration: How long to run the visualization (seconds)
    """
    
    # Setup environment
    setup_mountain_environment()
    
    # Save creature to URDF and load
    xml_file = 'temp_viz.urdf'
    xml_str = cr.to_xml()
    with open(xml_file, 'w') as f:
        f.write(xml_str)
    
    cid = p.loadURDF(xml_file)
    
    # Position creature on the flat ground, away from mountain (same as training)
    start_x = 6  # Start further away from mountain center
    start_y = 0
    start_z = 0.5  # Start just above ground level
    p.resetBasePositionAndOrientation(
        cid,
        [start_x, start_y, start_z],
        [0, 0, 0, 1]
    )
    
    # Set camera to view the scene
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 2]
    )
    
    # Run step simulation (not real-time for better control)
    print(f"\nVisualizing creature for {duration} seconds...")
    print("Press Ctrl+C to exit early.")
    print()
    
    max_height = 0
    step = 0
    total_steps = int(duration * 240)  # 240 fps
    motors = cr.get_motors()
    
    try:
        for step in range(total_steps):
            # Step the simulation
            p.stepSimulation()
            
            # Update motors every 24 steps (10 times per second simulation time)
            if step % 24 == 0:
                for jid in range(p.getNumJoints(cid)):
                    if jid < len(motors):
                        m = motors[jid]
                        p.setJointMotorControl2(
                            cid, jid,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=m.get_output(),
                            force=5  # Same as training
                        )
            
            # Get current position
            pos, _ = p.getBasePositionAndOrientation(cid)
            prev_max = max_height
            if pos[2] > max_height:
                max_height = pos[2]
                # Announce when new peak is reached
                if max_height > prev_max + 0.5:
                    print(f"\n*** NEW PEAK REACHED: {max_height:.3f} ***")
            
            # Print status every 240 steps (1 second)
            if step % 240 == 0:
                elapsed = step / 240.0
                h_dist = (pos[0]**2 + pos[1]**2)**0.5
                print(f"\rTime: {elapsed:.1f}s | Height: {pos[2]:.3f} | Dist to center: {h_dist:.2f} | Max: {max_height:.3f}", end="")
            
            # Small delay for visualization (slow down to ~real-time)
            time.sleep(1.0/240.0)
            
    except KeyboardInterrupt:
        pass
    
    print(f"\n\nFinal max height achieved: {max_height:.3f}")
    
    # Keep window open
    print("\nPress Enter to close...")
    input()
    p.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evolved creatures in the mountain environment"
    )
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="Path to creature CSV file")
    parser.add_argument("--random", action="store_true",
                        help="Generate and visualize a random creature")
    parser.add_argument("--gene_count", type=int, default=3,
                        help="Number of genes for random creature (default: 3)")
    parser.add_argument("--duration", type=int, default=20,
                        help="Visualization duration in seconds (default: 20)")
    
    args = parser.parse_args()
    
    if args.random:
        print("Generating random creature...")
        cr = creature.Creature(gene_count=args.gene_count)
    elif args.csv_file:
        print(f"Loading creature from: {args.csv_file}")
        cr = load_creature_from_csv(args.csv_file)
    else:
        parser.print_help()
        print("\nError: Please provide a CSV file or use --random")
        sys.exit(1)
    
    print(f"Creature has {len(cr.get_expanded_links())} links")
    
    visualize_creature(cr, duration=args.duration)


if __name__ == "__main__":
    main()

