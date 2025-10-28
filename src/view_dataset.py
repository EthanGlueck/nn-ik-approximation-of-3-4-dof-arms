import numpy as np
import pandas as pd
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_and_view_dataset(filename="ik_dataset_3dof.npz"):
    """Load and display dataset information"""
    try:
       
        data_path = os.path.join(BASE_DIR, "models", "data", filename)
        data = np.load(data_path)
        X = data['X']  # tip positions
        Y = data['Y']  # joint angles
        
        print("Dataset Info:")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Input shape (tip positions): {X.shape}")
        print(f"Output shape (joint angles): {Y.shape}")
        
        print("\n" + "="*50)
        print("Dataset Statistics:")
        print("="*50)
        
        print("\nTip Position Ranges:")
        print(f"  X: {X[:, 0].min():.3f} to {X[:, 0].max():.3f}")
        print(f"  Y: {X[:, 1].min():.3f} to {X[:, 1].max():.3f}")
        print(f"  Z: {X[:, 2].min():.3f} to {X[:, 2].max():.3f}")
        
        print("\nJoint Angle Ranges:")
        for i in range(Y.shape[1]):
            print(f"  Joint {i}: {Y[:, i].min():.3f} to {Y[:, i].max():.3f}")
        
        return X, Y
        
    except FileNotFoundError:
        print(f"Dataset file '{filename}' not found in models/data/ folder!")
        return None, None

def convert_to_excel(filename="ik_dataset_3dof.npz", output_name="ik_dataset.xlsx"):
    """Convert dataset to Excel format"""
    try:
        
        data_path = os.path.join(BASE_DIR, "models", "data", filename)
        data = np.load(data_path)
        X = data['X']
        Y = data['Y']
        
        # Dynamically create column names based on actual data shape
        tip_cols = ['tip_x', 'tip_y', 'tip_z']
        joint_cols = [f'joint_{i}' for i in range(Y.shape[1])]
        all_cols = tip_cols + joint_cols
        
        # Combine the data
        combined_data = np.column_stack([X, Y])
        
        # Create DataFrame with correct number of columns
        df = pd.DataFrame(combined_data, columns=all_cols)
        
        # Save Excel to current directory (or specify different path if needed)
        df.to_excel(output_name, index=False)
        print(f"Dataset converted to Excel: {output_name}")
        print(f"Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error converting to Excel: {e}")
        return None

if __name__ == "__main__":
    print("Dataset Viewer and Converter")
    print("="*40)
    
    # Ask which dataset to view
    print("Available datasets:")
    print("1. 3DOF dataset (ik_dataset_3dof.npz)")
    print("2. 4DOF dataset (ik_dataset_4dof.npz)")
    
    choice = input("Choose dataset (1 or 2): ").strip()
    
    if choice == "1":
        filename = "ik_dataset_3dof.npz"
        excel_name = "ik_dataset_3dof.xlsx"
    elif choice == "2":
        filename = "ik_dataset_4dof.npz"
        excel_name = "ik_dataset_4dof.xlsx"
    else:
        print("Invalid choice, using 3DOF dataset")
        filename = "ik_dataset_3dof.npz"
        excel_name = "ik_dataset_3dof.xlsx"
    
    # Load and view dataset
    X, Y = load_and_view_dataset(filename)
    
    if X is not None:
        # Ask user if they want to convert to Excel
        choice = input("\nConvert to Excel? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            convert_to_excel(filename, excel_name)
        else:
            print("Done viewing!")