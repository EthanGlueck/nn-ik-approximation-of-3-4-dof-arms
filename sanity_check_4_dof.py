import torch
import numpy as np
import pybullet as p
import pybullet_data
from forward_kinematics_4_dof import fk_4dof_batch

def compare_single_sample():
    """Compare your FK vs PyBullet FK on a single known joint configuration"""
    
    # Test with simple joint angles
    test_joints = np.array([0.0, 0.0, 0.0, 0.0])  # All joints at zero
    
    # Your FK result
    joints_tensor = torch.tensor(test_joints).unsqueeze(0)
    your_fk = fk_4dof_batch(joints_tensor).squeeze(0).numpy()
    
    # PyBullet FK result - GET THE TIP, NOT JUST LINK FRAME
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotId = p.loadURDF("4_dof_visual_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Set joints to test angles
    for i, angle in enumerate(test_joints):
        p.resetJointState(robotId, i, angle)
    
    # Get the tip link directly (link 4 is ee_tip for 4DOF)
    linkState = p.getLinkState(robotId, 4, computeForwardKinematics=1)
    pybullet_fk = np.array(linkState[4])
    
    p.disconnect()
    
    print("🔍 Single sample comparison (all joints = 0°):")
    print(f"Your FK:    ({your_fk[0]:.4f}, {your_fk[1]:.4f}, {your_fk[2]:.4f})")
    print(f"PyBullet:   ({pybullet_fk[0]:.4f}, {pybullet_fk[1]:.4f}, {pybullet_fk[2]:.4f})")
    print(f"Difference: {np.linalg.norm(your_fk - pybullet_fk)*1000:.2f} mm")
    
    # Test with shoulder = 90°
    test_joints2 = np.array([0.0, 1.57, 0.0, 0.0])  # Shoulder at 90°
    
    joints_tensor2 = torch.tensor(test_joints2).unsqueeze(0)
    your_fk2 = fk_4dof_batch(joints_tensor2).squeeze(0).numpy()
    
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotId = p.loadURDF("4_dof_visual_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    for i, angle in enumerate(test_joints2):
        p.resetJointState(robotId, i, angle)
    
    linkState2 = p.getLinkState(robotId, 4, computeForwardKinematics=1)
    pybullet_fk2 = np.array(linkState2[4])
    
    p.disconnect()
    
    print(f"\n🔍 Single sample comparison (shoulder = 90°):")
    print(f"Your FK:    ({your_fk2[0]:.4f}, {your_fk2[1]:.4f}, {your_fk2[2]:.4f})")
    print(f"PyBullet:   ({pybullet_fk2[0]:.4f}, {pybullet_fk2[1]:.4f}, {pybullet_fk2[2]:.4f})")
    print(f"Difference: {np.linalg.norm(your_fk2 - pybullet_fk2)*1000:.2f} mm")
    
    # Test with wrist = 90° (new test for 4DOF)
    test_joints3 = np.array([0.0, 0.0, 0.0, 1.57])  # Wrist at 90°
    
    joints_tensor3 = torch.tensor(test_joints3).unsqueeze(0)
    your_fk3 = fk_4dof_batch(joints_tensor3).squeeze(0).numpy()
    
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robotId = p.loadURDF("4_dof_visual_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    for i, angle in enumerate(test_joints3):
        p.resetJointState(robotId, i, angle)
    
    linkState3 = p.getLinkState(robotId, 4, computeForwardKinematics=1)
    pybullet_fk3 = np.array(linkState3[4])
    
    p.disconnect()
    
    print(f"\n🔍 Single sample comparison (wrist = 90°):")
    print(f"Your FK:    ({your_fk3[0]:.4f}, {your_fk3[1]:.4f}, {your_fk3[2]:.4f})")
    print(f"PyBullet:   ({pybullet_fk3[0]:.4f}, {pybullet_fk3[1]:.4f}, {pybullet_fk3[2]:.4f})")
    print(f"Difference: {np.linalg.norm(your_fk3 - pybullet_fk3)*1000:.2f} mm")

compare_single_sample()

# Load your dataset
data = np.load("ik_dataset_4dof.npz")
X = data["X"]  # PyBullet tip positions (normalized)
Y = data["Y"]  # Joint angles (normalized)

# Denormalize joint angles to radians (same as your training/testing)
Y_radians = Y * np.array([3.14, 1.57, 2.0, 2.0])  # Convert from [-1,1] to radians

# Denormalize PyBullet positions to meters
X_meters = X * np.array([0.94, 0.94, 0.6])  # Convert from [-1,1] to meters

# Use your actual FK function (first 10k samples for speed)
Y_tensor = torch.tensor(Y_radians[:10000], dtype=torch.float32)
fk_from_gt = fk_4dof_batch(Y_tensor).numpy()   # Use your FK file

pyb_positions = X_meters[:10000]
errors = np.linalg.norm(fk_from_gt - pyb_positions, axis=1)

print("FK vs PyBullet comparison:")
print(f"  Mean error:  {errors.mean()*1000:.2f} mm")
print(f"  Std error:   {errors.std()*1000:.2f} mm") 
print(f"  Max error:   {errors.max()*1000:.2f} mm")
print(f"  Min error:   {errors.min()*1000:.2f} mm")

# Show some sample comparisons
print(f"\nSample comparisons (first 5):")
for i in range(5):
    fk_pos = fk_from_gt[i]
    pyb_pos = pyb_positions[i]
    error_mm = np.linalg.norm(fk_pos - pyb_pos) * 1000
    
    print(f"Sample {i+1}: Error {error_mm:.2f} mm")
    print(f"  Your FK:  ({fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f})")
    print(f"  PyBullet: ({pyb_pos[0]:.4f}, {pyb_pos[1]:.4f}, {pyb_pos[2]:.4f})")