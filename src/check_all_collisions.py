import pybullet as p
import random

def check_ground_collision(robotId, joint_values, numJoints):
    """Check if robot collides with ground"""
    original_positions = [p.getJointState(robotId, i)[0] for i in range(numJoints)]
    
    for i, value in enumerate(joint_values):
        p.resetJointState(robotId, i, value)
    p.stepSimulation()
    
    contacts = p.getContactPoints(bodyA=robotId, bodyB=1)
    has_collision = len(contacts) > 0
    
    for i, pos in enumerate(original_positions):
        p.resetJointState(robotId, i, pos)
    
    return has_collision

def check_self_collision(robotId, joint_values, numJoints):
    """Check if robot collides with itself"""
    original_positions = [p.getJointState(robotId, i)[0] for i in range(numJoints)]
    
    for i, value in enumerate(joint_values):
        p.resetJointState(robotId, i, value)
    p.stepSimulation()
    
    # Check for contacts between different links of the same robot
    contacts = p.getContactPoints(bodyA=robotId, bodyB=robotId)
    has_self_collision = len(contacts) > 0
    
    for i, pos in enumerate(original_positions):
        p.resetJointState(robotId, i, pos)
    
    return has_self_collision

def check_all_collisions(robotId, joint_values, numJoints):
    """
    Check if the given joint configuration results in any collisions.
    Returns True if there are collisions, False otherwise.
    """
    # Set the robot to the given joint configuration
    for i in range(numJoints):  # Use numJoints parameter instead of hardcoded 5
        p.resetJointState(robotId, i, joint_values[i])
    
    # Step the simulation to update positions
    p.stepSimulation()
    
    # Check for collisions between all pairs of links
    for i in range(numJoints):
        for j in range(i + 2, numJoints):  # Skip adjacent links
            contact_points = p.getClosestPoints(robotId, robotId, distance=0.01, linkIndexA=i, linkIndexB=j)
            if len(contact_points) > 0:
                return True
    
    # Check for collisions with the ground plane
    ground_contacts = p.getContactPoints(robotId, 0)  # 0 is typically the ground plane
    if len(ground_contacts) > 0:
        return True
    
    return False

def generate_collision_free_random_values(robotId, numJoints, current_values):
    """
    Generate random joint values that don't result in collisions.
    Uses the numJoints parameter to determine how many joints to generate.
    """
    # Define joint limits for different robot configurations
    if numJoints == 3:
        # 3DOF joint limits
        joint_limits = [
            (-3.14, 3.14),   # Waist joint
            (-1.57, 1.57),   # Shoulder joint  
            (-2.0, 2.0)      # Elbow joint
        ]
    elif numJoints == 4:
        # 4DOF joint limits
        joint_limits = [
            (-3.14, 3.14),   # Waist joint
            (-1.57, 1.57),   # Shoulder joint  
            (-2.62, 2.62),   # Elbow joint
            (-2.62, 2.62)    # Elbow2 joint
        ]
    elif numJoints == 5:
        # 5DOF joint limits
        joint_limits = [
            (-3.14, 3.14),   # Base joint
            (-1.57, 1.57),   # Shoulder joint
            (-2.62, 2.62),   # Elbow joint
            (-2.62, 2.62),   # Elbow2 joint
            (-2.62, 2.62)    # Wrist1 joint
        ]
    else:
        # Default limits for any number of joints
        joint_limits = [(-3.14, 3.14)] * numJoints
    
    max_attempts = 100
    
    for attempt in range(max_attempts):
        # Generate random values within joint limits
        candidate_values = []
        for i in range(numJoints):
            min_val, max_val = joint_limits[i]
            candidate_values.append(random.uniform(min_val, max_val))
        
        # Check if this configuration is collision-free
        if not check_all_collisions(robotId, candidate_values, numJoints):
            return candidate_values
    
    # If we couldn't find a collision-free configuration, return current values
    print(f"Warning: Could not find collision-free configuration after {max_attempts} attempts")
    return current_values[:numJoints]  # Return only the number of joints needed