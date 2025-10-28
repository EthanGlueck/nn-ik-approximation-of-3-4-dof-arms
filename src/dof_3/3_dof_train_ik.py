import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import csv
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forward_kinematics_3_dof import fk_3dof_batch, fk_3dof_single, ROBOT_PARAMS, get_workspace_limits


def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimize for GPU performance
    torch.backends.cudnn.benchmark = True

    # Load 3DOF dataset (already normalized)
    data_path = os.path.join(BASE_DIR, "models", "data", "ik_dataset_3dof.npz")
    data = np.load(data_path)
    X = data["X"]  # tip positions (already normalized)
    Y = data["Y"]  # joint angles (already normalized)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    print("📊 Dataset loaded:")
    print(f"Dataset shapes: X={X_tensor.shape}, Y={Y_tensor.shape}")
    print(f"X range: [{X_tensor.min():.3f}, {X_tensor.max():.3f}]")
    print(f"Y range: [{Y_tensor.min():.3f}, {Y_tensor.max():.3f}]")

    # Dataset & split - using already normalized data
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Use fewer workers to avoid Windows multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)

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
    
    net = IKNet3DOF.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )
    
    num_epochs = 30

    print(f"\n🚀 Training with Position-Based Loss:")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Training loop with position-based loss
    loss_history = []
    prev_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        train_samples = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True) 
            
            optimizer.zero_grad()
            
            # Predict normalized joint angles
            Y_pred = net(X_batch)
            
            # Denormalize predicted joints for FK (convert from [-1,1] to radians)
            Y_pred_radians = torch.stack([
                Y_pred[:, 0] * 3.14,  # Waist: [-1,1] → [-3.14, 3.14]
                Y_pred[:, 1] * 1.57,  # Shoulder: [-1,1] → [-1.57, 1.57]
                Y_pred[:, 2] * 2.0    # Elbow: [-1,1] → [-2.0, 2.0]
            ], dim=1)
            
            # Forward kinematics: convert predicted joints to tip positions
            tip_pred = fk_3dof_batch(Y_pred_radians)
            
            # Denormalize target tip positions for comparison (convert from [-1,1] to meters)
            X_batch_meters = torch.stack([
                X_batch[:, 0] * 0.37,  # X: [-1,1] → [-0.37, 0.37]
                X_batch[:, 1] * 0.37,  # Y: [-1,1] → [-0.37, 0.37]  
                X_batch[:, 2] * 0.5    # Z: [-1,1] → [0, 0.5]
            ], dim=1)
            
            # Position-based loss: compare predicted tip vs actual tip (both in meters)
            position_loss = nn.MSELoss()(tip_pred, X_batch_meters)
            
            # Optional: Add small joint regularization to avoid extreme solutions
            joint_reg = 0.01 * nn.MSELoss()(Y_pred, Y_batch)
            
            total_loss = position_loss + joint_reg
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * X_batch.size(0)
            train_samples += X_batch.size(0)
        
        train_loss /= train_samples

        # Evaluate on test set
        net.eval()
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                
                Y_pred = net(X_batch)
                
                # Denormalize for FK
                Y_pred_radians = torch.stack([
                    Y_pred[:, 0] * 3.14,
                    Y_pred[:, 1] * 1.57,
                    Y_pred[:, 2] * 2.0
                ], dim=1)
                
                tip_pred = fk_3dof_batch(Y_pred_radians)
                
                # Denormalize target positions
                X_batch_meters = torch.stack([
                    X_batch[:, 0] * 0.37,
                    X_batch[:, 1] * 0.37,
                    X_batch[:, 2] * 0.5
                ], dim=1)
                
                position_loss = nn.MSELoss()(tip_pred, X_batch_meters)
                joint_reg = 0.01 * nn.MSELoss()(Y_pred, Y_batch)
                
                total_loss = position_loss + joint_reg
                test_loss += total_loss.item() * X_batch.size(0)
                test_samples += X_batch.size(0)
        
        test_loss /= test_samples
        
        # Step scheduler
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr != prev_lr:
            print(f"📉 Learning rate reduced to {current_lr:.2e}")
            prev_lr = current_lr

        # Store loss data
        loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'learning_rate': current_lr
        })

        # Print progress
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, LR: {current_lr:.2e}")

    # Save everything to proper locations
    loss_csv_path = os.path.join(BASE_DIR, "models", "data", "training_loss_3dof.csv")
    with open(loss_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'test_loss', 'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    # Save model 
    model_path = os.path.join(BASE_DIR, "models", "iknet_model_3dof.pth")
    torch.save(net.state_dict(), model_path)

    print(" Position-based 3DOF Model saved!")
    print(f" Loss history saved to '{loss_csv_path}'")

    # Test the trained model
    print("\n Testing Position-Based Model:")
    net.eval()
    
    with torch.no_grad():
        test_sample = next(iter(test_loader))
        X_batch, Y_batch = test_sample
        X_batch = X_batch[:5].to(device)
        
        Y_pred = net(X_batch)
        
        # Denormalize for FK
        Y_pred_radians = torch.stack([
            Y_pred[:, 0] * 3.14,
            Y_pred[:, 1] * 1.57,
            Y_pred[:, 2] * 2.0
        ], dim=1)
        
        tip_pred = fk_3dof_batch(Y_pred_radians)
        
        # Denormalize target positions
        X_batch_meters = torch.stack([
            X_batch[:, 0] * 0.37,
            X_batch[:, 1] * 0.37,
            X_batch[:, 2] * 0.5
        ], dim=1)
        
        print("Position Errors (mm):")
        for i in range(5):
            target = X_batch_meters[i].cpu().numpy()
            predicted = tip_pred[i].cpu().numpy()
            error = np.linalg.norm(predicted - target) * 1000
            
            print(f"Sample {i+1}: {error:.1f} mm")
            print(f"  Target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
            print(f"  Predicted: ({predicted[0]:.3f}, {predicted[1]:.3f}, {predicted[2]:.3f})")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()