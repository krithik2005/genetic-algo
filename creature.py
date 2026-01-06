import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class SensorType(Enum):
    RAYCAST = 0      # Detect distance to mountain
    ORIENTATION = 1  # Detect orientation relative to mountain
    CONTACT = 2      # Detect ground contact
    HEIGHT = 3       # Detect current height

class Sensor:
    """Advanced sensor system for reactive control - exceptional criteria"""
    def __init__(self, sensor_type, sensitivity, direction):
        self.sensor_type = sensor_type
        self.sensitivity = sensitivity
        self.direction = np.array(direction)
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        self.value = 0.0
    
    def read(self, creature_position, creature_orientation, mountain_center=(0, 0, 0), physics_client=None, creature_id=None):
        """Read sensor value based on creature state"""
        if self.sensor_type == SensorType.RAYCAST:
            # Calculate distance to mountain center (horizontal)
            h_dist = np.sqrt((creature_position[0] - mountain_center[0])**2 + 
                           (creature_position[1] - mountain_center[1])**2)
            # Closer = higher value (inverse distance)
            self.value = max(0, 1.0 - h_dist / 10.0) * self.sensitivity
            return self.value
            
        elif self.sensor_type == SensorType.ORIENTATION:
            # Check if facing toward mountain
            to_mountain = np.array([mountain_center[0] - creature_position[0],
                                  mountain_center[1] - creature_position[1],
                                  0])
            if np.linalg.norm(to_mountain) > 0:
                to_mountain = to_mountain / np.linalg.norm(to_mountain)
                # Dot product with forward direction (simplified)
                alignment = np.dot(self.direction[:2], to_mountain[:2])
                self.value = max(0, alignment) * self.sensitivity
            return self.value
            
        elif self.sensor_type == SensorType.CONTACT:
            # Simplified: check if near ground (z close to 0)
            if creature_position[2] < 1.0:
                self.value = 1.0 * self.sensitivity
            else:
                self.value = 0.0
            return self.value
            
        elif self.sensor_type == SensorType.HEIGHT:
            # Current height normalized
            self.value = (creature_position[2] / 10.0) * self.sensitivity
            return self.value
        
        return 0.0

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, sensor=None):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0
        self.sensor = sensor  # Advanced: sensor input for reactive control
    

    def get_output(self, sensor_value=0.0):
        """Get motor output, optionally modulated by sensor input"""
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                base_output = 1
            else:
                base_output = -1
        else:  # SINE
            base_output = np.sin(self.phase)
        
        # Advanced: Sensor modulation
        if self.sensor is not None and sensor_value > 0:
            # Sensor can turn motor on/off or modulate amplitude
            # If sensor value is high, motor is more active
            modulated_output = base_output * (0.5 + 0.5 * sensor_value)
            return modulated_output * self.amp
        
        return base_output * self.amp 

class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.sensors = None  # Advanced: sensor system
        self.start_position = None
        self.last_position = None
        self.position_history = []  # Track position history for better fitness
        self.max_height = 0  # Track maximum height achieved for mountain climbing
        self.best_position = None  # Position where max height was achieved
        self.ever_out_of_bounds = False  # Track if creature ever left the arena box
        self.reached_center = False  # Track if creature reached mountain center
        self.closest_to_center = float('inf')  # Track closest distance to center
        self.mountain_peak_reached = 0.0  # Track how close to peak

    def get_flat_links(self):
        if self.flat_links == None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(self.flat_links[0], 
                                self.flat_links[0].name, 
                                self.flat_links, 
                                exp_links)
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.exp_links:
            if first:# skip the root node! 
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "pepe") #  choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_sensors(self):
        """Get sensors for each link - advanced feature"""
        self.get_expanded_links()
        if self.sensors == None:
            sensors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                # Determine sensor type from gene
                sensor_type_val = l.sensor_type
                if sensor_type_val < 0.25:
                    sensor_type = SensorType.RAYCAST
                elif sensor_type_val < 0.5:
                    sensor_type = SensorType.ORIENTATION
                elif sensor_type_val < 0.75:
                    sensor_type = SensorType.CONTACT
                else:
                    sensor_type = SensorType.HEIGHT
                
                direction = [l.sensor_direction_x, l.sensor_direction_y, l.sensor_direction_z]
                sensor = Sensor(sensor_type, l.sensor_sensitivity, direction)
                sensors.append(sensor)
            self.sensors = sensors
        return self.sensors
    
    def get_motors(self):
        self.get_expanded_links()
        if self.motors == None:
            motors = []
            sensors = self.get_sensors()
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                # Advanced: Link motor to sensor if sensor exists
                sensor = sensors[i-1] if i-1 < len(sensors) else None
                m = Motor(l.control_waveform, l.control_amp, l.control_freq, sensor=sensor)
                motors.append(m)
            self.motors = motors 
        return self.motors 
    
    def update_position(self, pos, arena_size=20):
        """
        Update creature position and track important metrics.
        
        Args:
            pos: Current position [x, y, z]
            arena_size: Size of the arena box (default 20, so bounds are -10 to +10)
        """
        if self.start_position == None:
            self.start_position = pos
        else:
            self.last_position = pos
        
        # Track position history for better fitness calculation
        self.position_history.append(pos)
        
        # Check if creature is within arena bounds (box)
        half_size = arena_size / 2.0
        in_bounds = (-half_size <= pos[0] <= half_size and 
                     -half_size <= pos[1] <= half_size)
        
        if not in_bounds:
            self.ever_out_of_bounds = True
        
        # Track distance to mountain center (0, 0)
        h_dist_to_center = np.sqrt(pos[0]**2 + pos[1]**2)
        if h_dist_to_center < self.closest_to_center:
            self.closest_to_center = h_dist_to_center
            if h_dist_to_center < 1.0:  # Within 1 unit of center
                self.reached_center = True
        
        # Track maximum height achieved (z-coordinate)
        # Only count if within bounds and reasonably close to mountain
        if in_bounds and pos[2] > self.max_height and h_dist_to_center < 8.0:
            self.max_height = pos[2]
            self.best_position = pos
            # Track how close to peak (mountain peak is at z=5 typically)
            if pos[2] > self.mountain_peak_reached:
                self.mountain_peak_reached = pos[2]

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        dist = np.linalg.norm(p1-p2)
        return dist 

    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.sensors = None
        self.start_position = None
        self.last_position = None
        self.position_history = []
        self.max_height = 0
        self.best_position = None
        self.ever_out_of_bounds = False
        self.reached_center = False
        self.closest_to_center = float('inf')
        self.mountain_peak_reached = 0.0

    def get_max_height(self):
        """Return the maximum height achieved by the creature."""
        return self.max_height
    
    def get_mountain_climbing_fitness(self, mountain_center=(0, 0, 0), mountain_peak_height=5.0, arena_size=20):
        """
        Fitness function that rewards creatures for:
        1. Reaching the center of the mountain (0, 0)
        2. Climbing to the top of the mountain
        3. Staying within the arena box bounds
        
        If creature goes outside the box, fitness is 0.
        """
        # CRITICAL: If creature ever left the arena box, no reward
        if self.ever_out_of_bounds:
            return 0.0
        
        if self.max_height == 0:
            return 0.0
        
        # Component 1: Reaching mountain center (0, 0) - BIG REWARD
        center_score = 0.0
        if self.reached_center:
            center_score = 1.0  # Full reward for reaching center
        else:
            # Partial reward based on how close to center
            # Closer to center = higher score (inverse distance)
            if self.closest_to_center < float('inf'):
                center_score = max(0, 1.0 - (self.closest_to_center / 10.0))
        
        # Component 2: Climbing to top of mountain - BIG REWARD
        height_score = min(self.mountain_peak_reached / mountain_peak_height, 1.0)
        
        # Component 3: Combined position score (at center AND high up)
        # Best position is at center (0,0) and at peak height
        position_score = 0.0
        if self.best_position:
            h_dist = np.sqrt(self.best_position[0]**2 + self.best_position[1]**2)
            # Reward being at center (0,0) with high z
            # If at center (h_dist < 1) and high up, maximum reward
            if h_dist < 1.0 and self.best_position[2] > mountain_peak_height * 0.8:
                position_score = 1.0
            else:
                # Partial reward: closer to center + higher = better
                center_proximity = max(0, 1.0 - (h_dist / 10.0))
                height_proximity = min(1.0, self.best_position[2] / mountain_peak_height)
                position_score = (center_proximity + height_proximity) / 2.0
        
        # Weighted combination:
        # - 40% for reaching/getting close to center
        # - 40% for climbing to top
        # - 20% for combined position (center + top)
        fitness = (0.4 * center_score + 
                   0.4 * height_score + 
                   0.2 * position_score) * 10.0  # Scale to 0-10 range
        
        return fitness
    
    def get_fitness_score(self, mountain_center=(0, 0, 0)):
        """Alias for compatibility"""
        return self.get_mountain_climbing_fitness(mountain_center)