"""
Mountain Climbing Simulation for CM3020 AI Coursework
This module provides simulation classes for evolving creatures to climb mountains.
"""

import pybullet as p
import pybullet_data
import os
from multiprocessing import Pool


class MountainSimulation:
    """
    Simulation class for the mountain climbing environment.
    Creatures are evaluated based on their ability to climb the mountain (max height achieved).
    Advanced: Supports multiple landscape types for generalization testing.
    """
    
    def __init__(self, sim_id=0, arena_size=20, mountain_height=5, landscape_type="gaussian_pyramid"):
        self.physicsClientId = p.connect(p.DIRECT)
        self.sim_id = sim_id
        self.arena_size = arena_size
        self.mountain_height = mountain_height
        self.landscape_type = landscape_type  # Advanced: support different landscapes
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.shapes_dir = os.path.join(self.script_dir, 'shapes')
    
    def _setup_environment(self):
        """Set up the mountain climbing environment with arena and mountain."""
        pid = self.physicsClientId
        
        # Set gravity
        p.setGravity(0, 0, -10, physicsClientId=pid)
        
        # Create arena floor
        wall_thickness = 0.5
        floor_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX, 
            halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness],
            physicsClientId=pid
        )
        floor_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness],
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=pid
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0, 0, -wall_thickness],
            physicsClientId=pid
        )
        
        # Create arena walls
        wall_height = 1
        # Front and back walls
        wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, wall_thickness/2, wall_height/2],
            physicsClientId=pid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         basePosition=[0, self.arena_size/2, wall_height/2], physicsClientId=pid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         basePosition=[0, -self.arena_size/2, wall_height/2], physicsClientId=pid)
        
        # Left and right walls
        wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness/2, self.arena_size/2, wall_height/2],
            physicsClientId=pid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         basePosition=[self.arena_size/2, 0, wall_height/2], physicsClientId=pid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         basePosition=[-self.arena_size/2, 0, wall_height/2], physicsClientId=pid)
        
        # Load the mountain - Advanced: support different landscape types
        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        p.setAdditionalSearchPath(self.shapes_dir, physicsClientId=pid)
        
        # Map landscape type to URDF file
        landscape_map = {
            "gaussian_pyramid": "gaussian_pyramid.urdf",
            "mountain_with_cubes": "mountain_with_cubes.urdf",
            "mountain": "mountain.urdf"
        }
        
        urdf_file = landscape_map.get(self.landscape_type, "gaussian_pyramid.urdf")
        mountain = p.loadURDF(
            urdf_file,
            mountain_position,
            mountain_orientation,
            useFixedBase=1,
            physicsClientId=pid
        )
        
        return mountain

    def run_creature(self, cr, iterations=2400):
        """
        Run a single creature in the mountain environment and evaluate its performance.
        
        Args:
            cr: Creature object to evaluate
            iterations: Number of simulation steps (at 240fps)
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)
        
        # Set up the mountain environment
        self._setup_environment()
        
        # Save creature to URDF and load into simulation
        xml_file = os.path.join(self.script_dir, 'temp_mountain_' + str(self.sim_id) + '.urdf')
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        cid = p.loadURDF(xml_file, physicsClientId=pid)
        
        # Position creature on the flat ground, away from mountain
        # Creature must crawl/move toward the mountain to climb it
        start_x = 6  # Start further away from mountain center
        start_y = 0
        start_z = 0.5  # Start just above ground level (not high up)
        p.resetBasePositionAndOrientation(
            cid, 
            [start_x, start_y, start_z], 
            [0, 0, 0, 1], 
            physicsClientId=pid
        )
        
        # Run simulation
        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)
            
            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos, arena_size=self.arena_size)
    
    def update_motors(self, cid, cr):
        """
        Update motor velocities for the creature.
        Advanced: Supports sensor-based reactive control.
        
        Args:
            cid: Physics engine ID of the creature
            cr: Creature object
        """
        # Get current position and orientation for sensors
        pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=self.physicsClientId)
        mountain_center = (0, 0, 0)  # Mountain is at origin
        
        motors = cr.get_motors()
        sensors = cr.get_sensors() if hasattr(cr, 'get_sensors') else [None] * len(motors)
        
        for jid in range(p.getNumJoints(cid, physicsClientId=self.physicsClientId)):
            if jid < len(motors):
                m = motors[jid]
                # Advanced: Read sensor value if sensor exists
                sensor_value = 0.0
                if jid < len(sensors) and sensors[jid] is not None:
                    sensor_value = sensors[jid].read(
                        pos, orn, 
                        mountain_center=mountain_center,
                        physics_client=self.physicsClientId,
                        creature_id=cid
                    )
                
                # Get motor output (sensor-modulated)
                velocity = m.get_output(sensor_value=sensor_value)
                
                p.setJointMotorControl2(
                    cid, jid,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=velocity,
                    force=5,
                    physicsClientId=self.physicsClientId
                )


class ThreadedMountainSim:
    """
    Multi-threaded version of the mountain simulation for faster evaluation.
    Advanced: Supports different landscape types.
    """
    
    def __init__(self, pool_size, arena_size=20, mountain_height=5, landscape_type="gaussian_pyramid"):
        self.sims = [
            MountainSimulation(i, arena_size, mountain_height, landscape_type) 
            for i in range(pool_size)
        ]
        self.pool_size = pool_size
        self.landscape_type = landscape_type
    
    @staticmethod
    def static_run_creature(sim, cr, iterations):
        """Static method for multiprocessing pool."""
        sim.run_creature(cr, iterations)
        return cr
    
    def eval_population(self, pop, iterations):
        """
        Evaluate all creatures in a population.
        
        Args:
            pop: Population object
            iterations: Number of simulation steps per creature
        """
        pool_args = []
        start_ind = 0
        pool_size = len(self.sims)
        
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):
                    break
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                    self.sims[sim_ind],
                    pop.creatures[i],
                    iterations
                ])
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size
        
        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                creatures = p.starmap(ThreadedMountainSim.static_run_creature, pool_argset)
                new_creatures.extend(creatures)
        
        pop.creatures = new_creatures


class GUIMountainSimulation:
    """
    GUI version of the mountain simulation for visualization.
    """
    
    def __init__(self, arena_size=20, mountain_height=5):
        self.physicsClientId = p.connect(p.GUI)
        self.arena_size = arena_size
        self.mountain_height = mountain_height
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.shapes_dir = os.path.join(self.script_dir, 'shapes')
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def setup_environment(self):
        """Set up the visible mountain environment."""
        pid = self.physicsClientId
        
        p.setGravity(0, 0, -10, physicsClientId=pid)
        
        # Create arena floor
        wall_thickness = 0.5
        floor_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness],
            physicsClientId=pid
        )
        floor_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness],
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=pid
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=floor_collision,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0, 0, -wall_thickness],
            physicsClientId=pid
        )
        
        # Create arena walls
        wall_height = 1
        wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, wall_thickness/2, wall_height/2],
            physicsClientId=pid
        )
        wall_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.arena_size/2, wall_thickness/2, wall_height/2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=pid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         baseVisualShapeIndex=wall_visual,
                         basePosition=[0, self.arena_size/2, wall_height/2], physicsClientId=pid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         baseVisualShapeIndex=wall_visual,
                         basePosition=[0, -self.arena_size/2, wall_height/2], physicsClientId=pid)
        
        wall_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness/2, self.arena_size/2, wall_height/2],
            physicsClientId=pid
        )
        wall_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[wall_thickness/2, self.arena_size/2, wall_height/2],
            rgbaColor=[0.7, 0.7, 0.7, 1],
            physicsClientId=pid
        )
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         baseVisualShapeIndex=wall_visual,
                         basePosition=[self.arena_size/2, 0, wall_height/2], physicsClientId=pid)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision,
                         baseVisualShapeIndex=wall_visual,
                         basePosition=[-self.arena_size/2, 0, wall_height/2], physicsClientId=pid)
        
        # Load the mountain
        mountain_position = (0, 0, -1)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        p.setAdditionalSearchPath(self.shapes_dir, physicsClientId=pid)
        mountain = p.loadURDF(
            "gaussian_pyramid.urdf",
            mountain_position,
            mountain_orientation,
            useFixedBase=1,
            physicsClientId=pid
        )
        
        return mountain
    
    def run_creature(self, cr, iterations=2400, realtime=True):
        """
        Run and visualize a creature in the mountain environment.
        
        Args:
            cr: Creature object
            iterations: Number of simulation steps
            realtime: If True, run in real-time for visualization
        """
        import time
        pid = self.physicsClientId
        
        # Save creature to URDF and load
        xml_file = os.path.join(self.script_dir, 'temp_gui.urdf')
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        cid = p.loadURDF(xml_file, physicsClientId=pid)
        
        # Position creature
        start_x = 4
        start_y = 0
        start_z = 2.5
        p.resetBasePositionAndOrientation(
            cid,
            [start_x, start_y, start_z],
            [0, 0, 0, 1],
            physicsClientId=pid
        )
        
        if realtime:
            p.setRealTimeSimulation(1, physicsClientId=pid)
            time.sleep(iterations / 240)  # Wait for simulation
        else:
            for step in range(iterations):
                p.stepSimulation(physicsClientId=pid)
                if step % 24 == 0:
                    for jid in range(p.getNumJoints(cid, physicsClientId=pid)):
                        m = cr.get_motors()[jid]
                        p.setJointMotorControl2(
                            cid, jid,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=m.get_output(),
                            force=5,
                            physicsClientId=pid
                        )
                pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
                cr.update_position(pos, arena_size=self.arena_size)
        
        return cr.get_max_height()
    
    def close(self):
        """Disconnect from the physics server."""
        p.disconnect(physicsClientId=self.physicsClientId)

