import torch
import pybullet as p
import pybullet_data
import time
import os
import sys
import contextlib
from train_iknet import IKNet

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load model and move to GPU
net = IKNet().to(device)
net.load_state_dict(torch.load("iknet_model.pth"))
net.eval()

# Initialize PyBullet with suppressed output
with suppress_output():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robotId = p.loadURDF("5_dof_visual_arm.urdf", basePosition=[0,0,0], useFixedBase=True)

tipLinkIndex = 4

# Example target tip position (normalized) - move to GPU
new_tip = torch.tensor([[0.2 / 0.8, 0.1 / 0.8, 2*(0.3 - 0)/0.93 - 1]], dtype=torch.float32).to(device)

# Predict sine/cosine values on GPU, then move back to CPU
with torch.no_grad():
    pred_sincos = net(new_tip).cpu()  # Shape: [1, 10]
    
    # Split sine and cosine values
    pred_sin = pred_sincos[:, :5]  # First 5 values are sine
    pred_cos = pred_sincos[:, 5:]  # Last 5 values are cosine
    
    # Convert back to angles using atan2 (handles all quadrants correctly)
    predicted_angles_rad = torch.atan2(pred_sin, pred_cos).numpy()[0]  # Shape: [5]

# The predicted angles are already in radians (denormalized)
# No need to multiply by normalization factors since we denormalized before sine/cosine
predicted_angles = predicted_angles_rad.tolist()

print(f"Predicted angles (radians): {[f'{angle:.3f}' for angle in predicted_angles]}")

# Set robot to predicted angles
for i, angle in enumerate(predicted_angles):
    p.resetJointState(robotId, i, angle)

p.setRealTimeSimulation(1)

# Print results once
tip_pos = p.getLinkState(robotId, tipLinkIndex)[0]
print(f"Target: X={0.2:.3f}, Y={0.1:.3f}, Z={0.3:.3f}")
print(f"Actual: X={tip_pos[0]:.3f}, Y={tip_pos[1]:.3f}, Z={tip_pos[2]:.3f}")
print(f"Error: X={abs(0.2-tip_pos[0]):.3f}, Y={abs(0.1-tip_pos[1]):.3f}, Z={abs(0.3-tip_pos[2]):.3f}")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    p.disconnect()
