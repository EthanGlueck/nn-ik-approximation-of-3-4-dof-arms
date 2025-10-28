import torch
import numpy as np

def fk_4dof_batch(joints_batch):
    """
    Batch 4DOF forward kinematics matching PyBullet URDF
    """
    # Link lengths from URDF
    L0 = 0.13   # base to shoulder height
    L1 = 0.2    # shoulder to elbow length  
    L2 = 0.15   # elbow to wrist length
    L3 = 0.12   # wrist to tip length

    waist = joints_batch[:, 0]
    shoulder = joints_batch[:, 1]
    elbow = joints_batch[:, 2]
    wrist = joints_batch[:, 3]

    # Build up the kinematic chain
    # 1. Shoulder joint position
    shoulder_x = torch.zeros_like(waist)
    shoulder_y = torch.zeros_like(waist)  
    shoulder_z = L0 * torch.ones_like(waist)
    
    # 2. Elbow position (shoulder rotation around X-axis)
    elbow_x = shoulder_x
    elbow_y = shoulder_y + L1 * torch.sin(shoulder)
    elbow_z = shoulder_z + L1 * torch.cos(shoulder)
    
    # 3. Wrist position (elbow rotation around X-axis)  
    wrist_x = elbow_x
    wrist_y = elbow_y + L2 * torch.sin(shoulder + elbow)
    wrist_z = elbow_z + L2 * torch.cos(shoulder + elbow)
    
    # 4. Wrist rotation around X-axis (same as shoulder/elbow)
    # All three joints (shoulder, elbow, wrist) now rotate around X-axis
    combined_angle = shoulder + elbow + wrist
    
    tip_x = wrist_x
    tip_y = wrist_y + L3 * torch.sin(combined_angle)
    tip_z = wrist_z + L3 * torch.cos(combined_angle)
    
    # Waist rotation around Z-axis
    waist_corrected = waist + 3.14159
    cos_w, sin_w = torch.cos(waist_corrected), torch.sin(waist_corrected)
    
    x_final = cos_w * tip_x - sin_w * tip_y
    y_final = sin_w * tip_x + cos_w * tip_y
    z_final = tip_z
    
    return torch.stack([x_final, y_final, z_final], dim=1)

def fk_4dof_single(joints):
    """
    Single sample 4DOF forward kinematics
    joints: [4] -> waist, shoulder, elbow, wrist (radians)
    Returns tip position [3] (meters)
    """
    if isinstance(joints, np.ndarray):
        joints = torch.tensor(joints, dtype=torch.float32)
    
    if len(joints.shape) == 1:
        joints = joints.unsqueeze(0)
    
    result = fk_4dof_batch(joints)
    return result.squeeze(0)

# Robot parameters for easy access
ROBOT_PARAMS = {
    'L0': 0.13,  # Base to shoulder height
    'L1': 0.2,   # Shoulder to elbow length  
    'L2': 0.15,  # Elbow to wrist length
    'L3': 0.12,  # Wrist to tip length
    'joint_limits': {
        'waist': (-3.14, 3.14),
        'shoulder': (-1.57, 1.57),
        'elbow': (-2.0, 2.0),
        'wrist': (-2.0, 2.0)
    },
    'normalization': {
        'tip_scale': [0.94, 0.94, 0.6],
        'joint_scale': [3.14, 1.57, 2.0, 2.0]
    }
}

def get_workspace_limits():
    """Get approximate workspace limits"""
    L0, L1, L2, L3 = ROBOT_PARAMS['L0'], ROBOT_PARAMS['L1'], ROBOT_PARAMS['L2'], ROBOT_PARAMS['L3']
    max_reach = L1 + L2 + L3
    min_reach = abs(L1 - L2 - L3)
    
    return {
        'x_range': (-max_reach, max_reach),
        'y_range': (-max_reach, max_reach), 
        'z_range': (L0 - max_reach, L0 + max_reach),
        'max_reach': max_reach,
        'min_reach': min_reach
    }