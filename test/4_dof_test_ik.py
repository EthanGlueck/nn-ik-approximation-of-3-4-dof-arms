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


from src.dof_4.forward_kinematics_4_dof import fk_4dof_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Load trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define network (same architecture as training)
    IKNet4DOF = nn.Sequential(
        nn.Linear(3, 512), 
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4)   # output: 4 joint angles
    )
    
    # Load trained weights
    model_path = os.path.join(BASE_DIR, "models", "iknet_model_4dof.pth")
    net = IKNet4DOF.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    return net, device

def visualize_multiple_predictions(samples_list):
    """Show arm cycling through multiple predictions"""
    print("🤖 Starting simulation with cycling examples...")
    
    # Start PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    # Load robot
    urdf_path = os.path.join(BASE_DIR, "src", "dof_4", "4_dof_visual_arm.urdf")
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
                p.addUserDebugText(f"Error: {error_mm:.1f} mm", [0.16, 0, 0.002], textColorRGB=[0, 0, 0], textSize=1.5)
                
                print(f"🔄 Switched to sample {current_sample + 1}/{len(samples_list)} - Error: {error_mm:.1f} mm")
            
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

def test_random_samples(num_samples=500):
    """Test model on random dataset samples"""
    print(f"🧪 Testing {num_samples} random samples from dataset")
    print("=" * 80)
    
    # Load model and data
    net, device = load_model()
    data_path = os.path.join(BASE_DIR, "models", "data", "ik_dataset_4dof.npz")
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
            Y_pred[:, 2] * 2.0,   # Elbow
            Y_pred[:, 3] * 2.0    # Wrist
        ], dim=1)
        
        # Forward kinematics to get actual tip positions
        tip_pred = fk_4dof_batch(Y_pred_radians)
        
        # Denormalize target positions
        X_meters = torch.stack([
            X_tensor[:, 0] * 0.94,  # X (4DOF reach)
            X_tensor[:, 1] * 0.94,  # Y (4DOF reach)
            X_tensor[:, 2] * 0.6    # Z (4DOF reach)
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
            print(f"  Generated angles:   ({pred_joints[0]:.3f}, {pred_joints[1]:.3f}, {pred_joints[2]:.3f}, {pred_joints[3]:.3f}) rad")
            print(f"  Generated angles:   ({np.degrees(pred_joints[0]):6.1f}°, {np.degrees(pred_joints[1]):6.1f}°, {np.degrees(pred_joints[2]):6.1f}°, {np.degrees(pred_joints[3]):6.1f}°)")
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
    
    # ✅ Save errors to CSV file for histogram analysis
    csv_filename = f"4dof_errors_{num_samples}_samples.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample_Index', 'Error_mm'])  # Header
        for i, error in enumerate(errors):
            writer.writerow([i+1, error])
    
    print(f"📄 Saved all {len(errors)} errors to {csv_filename}")
    
    # Summary based on ALL samples
    errors = np.array(errors)
    print(f"📊 Summary (based on all {num_samples} samples):")
    print(f"  Average error: {np.mean(errors):5.1f} mm")
    print(f"  Min error:     {np.min(errors):5.1f} mm")
    print(f"  Max error:     {np.max(errors):5.1f} mm")
    print(f"  Std deviation: {np.std(errors):5.1f} mm")
    
    # Show cycling simulation (only first 10 samples)
    print(f"\n🎯 Showing first 10 samples cycling every 5 seconds...")
    visualize_multiple_predictions(all_samples)

if __name__ == "__main__":
    test_random_samples(500)