import os
import pybullet as p
import pybullet_data
import time
import numpy as np
from check_all_collisions import check_all_collisions, generate_collision_free_random_values

# Dataset configuration
GENERATE_DATASET = True  # Set to False for interactive mode
USE_GUI = False  # Set to True to see the robot moving (slower)
TARGET_DATASET_SIZE = 1000000  

# Preallocated arrays
tip_positions = np.zeros((TARGET_DATASET_SIZE, 3), dtype=np.float32)
joint_angles = np.zeros((TARGET_DATASET_SIZE, 5), dtype=np.float32)
dataset_index = 0

def initialize_simulation():
    "Initialize the simulation"
    if USE_GUI:
        p.connect(p.GUI)
        print("🖥️ GUI mode - you can see the robot moving")
    else:
        p.connect(p.DIRECT)
        print("⚡ DIRECT mode - faster data generation (no visuals)")
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)
    
    robotId = p.loadURDF("5_dof_visual_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    if USE_GUI:
        p.setRealTimeSimulation(1)
    
    return robotId

def generate_and_store_pose(robotId, numJoints, index):
    "Generate a collision-free pose and store it"
    global tip_positions, joint_angles
    
    joint_values = generate_collision_free_random_values(robotId, numJoints, [0.0] * numJoints)
    
    for i, value in enumerate(joint_values):
        p.resetJointState(robotId, i, value)
    
    if not USE_GUI:
        p.stepSimulation()  # Only step manually in DIRECT mode
    
    tipLinkIndex = numJoints - 1
    linkState = p.getLinkState(robotId, tipLinkIndex)
    tip_pos = linkState[0]
    
    # Store normalized data directly in preallocated arrays
    tip_positions[index] = [
        tip_pos[0] / 0.8,
        tip_pos[1] / 0.8,
        2 * (tip_pos[2] - 0.0) / 0.93 - 1
    ]
    
    joint_angles[index] = [
        joint_values[0] / 3.14,
        joint_values[1] / 1.57,
        joint_values[2] / 2.62,
        joint_values[3] / 2.62,
        joint_values[4] / 2.62
    ]
    
    return tip_pos, joint_values

def save_dataset(current_size):
    """Save dataset to files"""
    if current_size == 0:
        print("No data to save")
        return
    
    # Only save the portion that has been filled
    X = tip_positions[:current_size]
    Y = joint_angles[:current_size]
    
    np.savez("ik_dataset.npz", X=X, Y=Y)
    print(f"Dataset saved: {current_size} samples")
    print(f"Input shape (tip positions): {X.shape}")
    print(f"Output shape (joint angles): {Y.shape}")

# Initialize simulation
robotId = initialize_simulation()
numJoints = 5

if GENERATE_DATASET:
    # Start timer
    start_time = time.time()
    
    print(f"🤖 Starting automatic dataset generation ({TARGET_DATASET_SIZE} samples)...")
    print(f"💾 Preallocated arrays: {tip_positions.nbytes + joint_angles.nbytes / 1024 / 1024:.1f} MB")
    
    if USE_GUI:
        print("Close the PyBullet window to stop early and save current data")
    else:
        print("Press Ctrl+C to stop early and save current data")
    
    try:
        while dataset_index < TARGET_DATASET_SIZE:
            tip_pos, joint_values = generate_and_store_pose(robotId, numJoints, dataset_index)
            dataset_index += 1
            
            if dataset_index % 30000 == 0:
                elapsed_time = time.time() - start_time
                samples_per_second = dataset_index / elapsed_time
                eta_seconds = (TARGET_DATASET_SIZE - dataset_index) / samples_per_second if samples_per_second > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"📊 Collected {dataset_index}/{TARGET_DATASET_SIZE} samples")
                print(f"   Last pose: tip=({tip_pos[0]:.2f}, {tip_pos[1]:.2f}, {tip_pos[2]:.2f})")
                print(f"   Speed: {samples_per_second:.1f} samples/sec, ETA: {eta_minutes:.1f} minutes")
            
            if dataset_index % 100000 == 0:
                save_dataset(dataset_index)
            
            # Only add delay in GUI mode
            if USE_GUI:
                time.sleep(0.001)
        
        # Final timer
        total_time = time.time() - start_time
        print("✅ Dataset generation complete!")
        print(f"⏱️ Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
        print(f"📈 Average speed: {TARGET_DATASET_SIZE/total_time:.1f} samples/second")
        save_dataset(dataset_index)
        
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        print(f"\n⚠️ Dataset generation interrupted after {total_time/60:.2f} minutes")
        print(f"📊 Collected {dataset_index} samples at {dataset_index/total_time:.1f} samples/second")
        save_dataset(dataset_index)
    
else:
    # Interactive mode (GUI required)
    if not USE_GUI:
        print("⚠️ Interactive mode requires GUI. Setting USE_GUI = True")
        p.disconnect()
        USE_GUI = True
        robotId = initialize_simulation()
    
    print("🎮 Interactive mode - use sliders and buttons")
    
    # Store current joint positions
    current_joint_positions = [0.0] * numJoints
    
    # Create GUI sliders for 5 joints
    joint1Slider = p.addUserDebugParameter("Base (Z)", -2.62, 2.62, 0)
    joint2Slider = p.addUserDebugParameter("Shoulder (X)", -1.57, 1.57, 0)
    joint3Slider = p.addUserDebugParameter("Elbow (X)", -2.62, 2.62, 0)
    joint4Slider = p.addUserDebugParameter("Elbow2 (Y)", -2.62, 2.62, 0)
    joint5Slider = p.addUserDebugParameter("Wrist1 (X)", -2.62, 2.62, 0)
    
    sliders = [joint1Slider, joint2Slider, joint3Slider, joint4Slider, joint5Slider]
    
    # Add buttons
    randomButton = p.addUserDebugParameter("Random Joints", 1, 0, 0)
    restartButton = p.addUserDebugParameter("Restart Simulation", 1, 0, 0)
    
    # Track button state
    lastButtonValue = 0
    lastRestartButtonValue = 0
    randomValues = [0, 0, 0, 0, 0]

    # Function to get tip position
    def get_tip_position():
        linkState = p.getLinkState(robotId, tipLinkIndex)
        return linkState[0]

    # Run in real-time mode
    p.setRealTimeSimulation(1)

    # Counter for periodic tip position printing
    printCounter = 0

    try:
        while True:
            # Check if restart button was pressed
            currentRestartButtonValue = p.readUserDebugParameter(restartButton)
            if currentRestartButtonValue != lastRestartButtonValue:
                print("Restarting simulation...")
                # Reset all joint positions to zero
                for i in range(numJoints):
                    p.resetJointState(robotId, i, 0)
                    current_joint_positions[i] = 0.0
                # Clear random values
                randomValues = [0, 0, 0, 0, 0]
                # Reset sliders to zero
                for slider in sliders:
                    p.removeUserDebugItem(slider)
                # Recreate sliders at zero position
                joint1Slider = p.addUserDebugParameter("Base (Z)", -3.14, 3.14, 0)
                joint2Slider = p.addUserDebugParameter("Shoulder (X)", -2.62, 2.62, 0)
                joint3Slider = p.addUserDebugParameter("Elbow (X)", -2.62, 2.62, 0)
                joint4Slider = p.addUserDebugParameter("Elbow2 (Y)", -2.62, 2.62, 0)
                joint5Slider = p.addUserDebugParameter("Wrist1 (X)", -2.62, 2.62, 0)
                sliders = [joint1Slider, joint2Slider, joint3Slider, joint4Slider, joint5Slider]
                # Reset counters
                printCounter = 0
                print("Simulation restarted - all joints and sliders reset to home position")
                lastRestartButtonValue = currentRestartButtonValue

            # Check if random button was pressed
            currentButtonValue = p.readUserDebugParameter(randomButton)
            if currentButtonValue != lastButtonValue:
                randomValues = generate_collision_free_random_values(robotId, numJoints, current_joint_positions)
                
                # Temporarily set joints to random values to calculate tip position
                for i, value in enumerate(randomValues):
                    p.resetJointState(robotId, i, value)
                
                # Get and print tip position with these random values
                randomTipPos = get_tip_position()
                print(f"Random tip position: X={randomTipPos[0]:.3f}, Y={randomTipPos[1]:.3f}, Z={randomTipPos[2]:.3f}")
                
                lastButtonValue = currentButtonValue

            # Read slider values and control joints
            for i, slider in enumerate(sliders):
                sliderPos = p.readUserDebugParameter(slider)
                if abs(sliderPos) < 0.1 and randomValues[i] != 0:
                    targetPos = randomValues[i]
                else:
                    targetPos = sliderPos
                    if abs(sliderPos) > 0.1:
                        randomValues[i] = 0
                
                current_joint_positions[i] = targetPos
                
                p.setJointMotorControl2(
                    bodyUniqueId=robotId,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=targetPos,
                    force=100
                )
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Simulation stopped")
    finally: p.disconnect()
