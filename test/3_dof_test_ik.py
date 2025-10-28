import numpy as np
import torch
import torch.nn as nn
import pybullet as p
import pybullet_data
import time
import csv  
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dof_3.forward_kinematics_3_dof import fk_3dof_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Load trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
     # Define the network for 3DOF 
    IKNet3DOF = nn.Sequential(
        nn.Linear(3, 512), 
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(), 
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 3)   # output: 3 joint angles
    )
    
    # Load trained weights
    model_path = os.path.join(BASE_DIR, "models", "iknet_model_3dof.pth")

    net = IKNet3DOF.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    return net, device

def visualize_prediction(target_pos, pred_joints, predicted_tip):
    """Show arm in simulation at predicted angles with target sphere"""
    print(" Starting simulation...")
    
    # Start PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Load robot
    urdf_path = os.path.join(BASE_DIR, "src", "dof_3", "3_dof_visual_arm.urdf")
    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Set joint positions to predicted angles
    for i, joint_angle in enumerate(pred_joints):
        p.setJointMotorControl2(
            bodyUniqueId=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angle,
            force=100
        )
    
    # Create target sphere (red) at original allowed point
    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[1, 0, 0, 0.8]
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=target_pos
    )
    
    # Create predicted tip sphere (green) 
    pred_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[0, 1, 0, 0.8]
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=pred_visual,
        basePosition=predicted_tip
    )
    
    # Add labels
    error_mm = np.linalg.norm(predicted_tip - target_pos) * 1000
    p.addUserDebugText("Target", target_pos + [0, 0, 0.025], textColorRGB=[1, 0, 0], textSize=1.2)
    p.addUserDebugText("Predicted", predicted_tip + [0, 0, 0.025], textColorRGB=[0, 1, 0], textSize=1.2)
    p.addUserDebugText(f"Error: {error_mm:.1f} mm", [0.16, 0, 0.002], textColorRGB=[0, 0, 0], textSize=1.5)
    
    print("Press 'q' to close simulation")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()

def visualize_multiple_predictions(samples_list):
    """Show arm cycling through multiple predictions"""
    print(" Starting simulation with cycling examples...")
    
    # Start PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Load robot
    urdf_path = os.path.join(BASE_DIR, "src", "dof_3", "3_dof_visual_arm.urdf")
    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Create visual objects that we'll reposition
    target_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[1, 0, 0, 0.8]
    )
    target_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=[0, 0, 0]
    )
    
    pred_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.015,
        rgbaColor=[0, 1, 0, 0.8]
    )
    pred_body = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=pred_visual,
        basePosition=[0, 0, 0]
    )
    
    current_sample = 0
    last_switch_time = time.time()
    switch_interval = 5.0  # 5 seconds per example
    
    print(f"Cycling through {len(samples_list)} examples every {switch_interval} seconds")
    print("Press 'q' to close simulation")
    
    # Keep simulation running
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to switch examples
            if current_time - last_switch_time >= switch_interval:
                current_sample = (current_sample + 1) % len(samples_list)
                last_switch_time = current_time
                
                sample = samples_list[current_sample]
                
                # Move robot to new joint positions
                for i, joint_angle in enumerate(sample['joints']):
                    p.setJointMotorControl2(
                        bodyUniqueId=robotId,
                        jointIndex=i,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_angle,
                        force=100
                    )
                
                # Move spheres to new positions
                p.resetBasePositionAndOrientation(target_body, sample['target'], [0, 0, 0, 1])
                p.resetBasePositionAndOrientation(pred_body, sample['predicted'], [0, 0, 0, 1])
                
                # Remove old debug text
                p.removeAllUserDebugItems()
                
                # Add new labels
                error_mm = sample['error']
                p.addUserDebugText("Target", sample['target'] + [0, 0, 0.025], textColorRGB=[1, 0, 0], textSize=1.2)
                p.addUserDebugText("Predicted", sample['predicted'] + [0, 0, 0.025], textColorRGB=[0, 1, 0], textSize=1.2)
                p.addUserDebugText(f"Sample {current_sample + 1}/{len(samples_list)}", [0.16, 0.0, 0.08], textColorRGB=[1, 0.5, 1], textSize=1.5)
                p.addUserDebugText(f"Error: {error_mm:.1f} mm", [0.16, 0, 0.002], textColorRGB=[1, 1, 0], textSize=1.5)
                
                print(f" Switched to sample {current_sample + 1}/{len(samples_list)} - Error: {error_mm:.1f} mm")
            
            p.stepSimulation()
            time.sleep(0.01)
            
            # Check for quit
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()

def test_random_samples(num_samples=500, show_simulation=False):
    """Test model on random dataset samples"""
    print(f" Testing {num_samples} random samples from dataset")
    print("=" * 80)
    
    # Load model and data
    net, device = load_model()
    data_path = os.path.join(BASE_DIR, "models", "data", "ik_dataset_3dof.npz")
    data = np.load(data_path)
    X = data["X"]  # tip positions (normalized)
    Y = data["Y"]  # joint angles (normalized)
    
    # Randomly sample from the dataset
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    X_samples = X[indices]
    Y_true = Y[indices]
    
    # Convert to tensors
    X_tensor = torch.tensor(X_samples, dtype=torch.float32).to(device)
    
    errors = []
    all_samples = []  # Store all samples for simulation
    
    with torch.no_grad():
        # Predict joint angles
        Y_pred = net(X_tensor)
        
        # Denormalize predicted joints for FK
        Y_pred_radians = torch.stack([
            Y_pred[:, 0] * 3.14,  # Waist
            Y_pred[:, 1] * 1.57,  # Shoulder
            Y_pred[:, 2] * 2.0    # Elbow
        ], dim=1)
        
        # Forward kinematics to get actual tip positions
        tip_pred = fk_3dof_batch(Y_pred_radians)
        
        # Denormalize target positions
        X_meters = torch.stack([
            X_tensor[:, 0] * 0.37,  # X
            X_tensor[:, 1] * 0.37,  # Y
            X_tensor[:, 2] * 0.5    # Z
        ], dim=1)
        
        # Print results for first 10 samples only
        print("Showing first 10 samples:")
        for i in range(min(10, num_samples)):  # Only print first 10
            target = X_meters[i].cpu().numpy()
            predicted = tip_pred[i].cpu().numpy()
            pred_joints = Y_pred_radians[i].cpu().numpy()
            
            error_mm = np.linalg.norm(predicted - target) * 1000
            
            print(f"Sample {i+1:2d}:")
            print(f"  Target position:    ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) m")
            print(f"  Generated angles:   ({pred_joints[0]:.3f}, {pred_joints[1]:.3f}, {pred_joints[2]:.3f}) rad")
            print(f"  Generated angles:   ({np.degrees(pred_joints[0]):6.1f}°, {np.degrees(pred_joints[1]):6.1f}°, {np.degrees(pred_joints[2]):6.1f}°)")
            print(f"  Actual tip (FK):    ({predicted[0]:.3f}, {predicted[1]:.3f}, {predicted[2]:.3f}) m")
            print(f"  Position error:     {error_mm:5.1f} mm")
            print()
        
        # Calculate errors for ALL samples (including the ones not printed)
        for i in range(num_samples):
            target = X_meters[i].cpu().numpy()
            predicted = tip_pred[i].cpu().numpy()
            pred_joints = Y_pred_radians[i].cpu().numpy()
            
            error_mm = np.linalg.norm(predicted - target) * 1000
            errors.append(error_mm)
            
            # Store sample for simulation (only first 10 for visualization)
            if i < 10:
                all_samples.append({
                    'target': target,
                    'predicted': predicted,
                    'joints': pred_joints,
                    'error': error_mm
                })
    
    #  Save errors to CSV file for histogram analysis
    csv_filename = f"3dof_errors_{num_samples}_samples.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample_Index', 'Error_mm'])  # Header
        for i, error in enumerate(errors):
            writer.writerow([i+1, error])
    
    print(f" Saved all {len(errors)} errors to {csv_filename}")
    
    # Summary based on ALL samples
    errors = np.array(errors)
    print(f" Summary (based on all {num_samples} samples):")
    print(f"  Average error: {np.mean(errors):5.1f} mm")
    print(f"  Min error:     {np.min(errors):5.1f} mm")
    print(f"  Max error:     {np.max(errors):5.1f} mm")
    print(f"  Std deviation: {np.std(errors):5.1f} mm")
    
    # Show simulation if requested (only first 10 samples)
    if show_simulation:
        print(f"\n Showing first 10 samples cycling every 5 seconds...")
        visualize_multiple_predictions(all_samples)

if __name__ == "__main__":
    # Ask for simulation option
    sim_choice = input("Show simulation example? (y/n): ").lower().strip()
    show_sim = sim_choice == 'y'
    
    test_random_samples(show_simulation=show_sim)