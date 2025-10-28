import torch
import numpy as np

def fk_3dof_batch(joints_batch):
    """
    Batch 3DOF forward kinematics matching PyBullet URDF
    joints_batch: [batch_size, 3] -> waist, shoulder, elbow (radians)
    Returns tip positions [batch_size, 3] (meters)
    """
    # Link lengths from URDF
    L0 = 0.13   # base to shoulder height (0.05 + 0.08)
    L1 = 0.2    # shoulder to elbow length
    L2 = 0.15   # elbow to tip length

    waist = joints_batch[:, 0]
    shoulder = joints_batch[:, 1]
    elbow = joints_batch[:, 2]

    # Start from shoulder joint position
    shoulder_x = torch.zeros_like(waist)
    shoulder_y = torch.zeros_like(waist)  
    shoulder_z = L0 * torch.ones_like(waist)
    
    # Shoulder rotation around X-axis (creates motion in Y-Z plane)
    elbow_x = shoulder_x  # No change in X
    elbow_y = shoulder_y + L1 * torch.sin(shoulder)  # Y motion  
    elbow_z = shoulder_z + L1 * torch.cos(shoulder)  # Z motion
    
    # Elbow rotation around X-axis (relative to current orientation)
    tip_x = elbow_x  # Still no change in X
    tip_y = elbow_y + L2 * torch.sin(shoulder + elbow)  # Y motion
    tip_z = elbow_z + L2 * torch.cos(shoulder + elbow)  # Z motion
    
    # ✅ FIX: Negate waist angle to correct 180° offset
    waist_corrected = waist + 3.14159  # Add 180°
    cos_w, sin_w = torch.cos(waist_corrected), torch.sin(waist_corrected)
    
    # Rotate the (X,Y) coordinates by corrected waist angle
    x_final = cos_w * tip_x - sin_w * tip_y
    y_final = sin_w * tip_x + cos_w * tip_y
    z_final = tip_z
    
    return torch.stack([x_final, y_final, z_final], dim=1)

def fk_3dof_single(joints):
    """
    Single sample 3DOF forward kinematics
    joints: [3] -> waist, shoulder, elbow (radians)
    Returns tip position [3] (meters)
    """
    if isinstance(joints, np.ndarray):
        joints = torch.tensor(joints, dtype=torch.float32)
    
    if len(joints.shape) == 1:
        joints = joints.unsqueeze(0)  # Add batch dimension
    
    result = fk_3dof_batch(joints)
    return result.squeeze(0)  # Remove batch dimension

# Robot parameters for easy access
ROBOT_PARAMS = {
    'L0': 0.13,  # Base to shoulder height
    'L1': 0.2,   # Shoulder to elbow length  
    'L2': 0.15,  # Elbow to tip length
    'joint_limits': {
        'waist': (-3.14, 3.14),
        'shoulder': (-1.57, 1.57),
        'elbow': (-2.0, 2.0)
    },
    'normalization': {
        'tip_scale': [0.37, 0.37, 0.5],
        'joint_scale': [3.14, 1.57, 2.0]
    }
}

def get_workspace_limits():
    """Get approximate workspace limits"""
    L0, L1, L2 = ROBOT_PARAMS['L0'], ROBOT_PARAMS['L1'], ROBOT_PARAMS['L2']
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    
    return {
        'x_range': (-max_reach, max_reach),
        'y_range': (-max_reach, max_reach), 
        'z_range': (L0 - max_reach, L0 + max_reach),
        'max_reach': max_reach,
        'min_reach': min_reach
    }