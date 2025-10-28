import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import csv

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimize for GPU performance
    torch.backends.cudnn.benchmark = True

    # Load dataset
    data = np.load("ik_dataset.npz")
    X = data["X"]  # tip positions
    Y = data["Y"]  # joint angles (normalized)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Denormalize joint angles
    Y_denorm = Y_tensor.clone()
    Y_denorm[:, 0] = Y_tensor[:, 0] * 3.14   # Base joint
    Y_denorm[:, 1] = Y_tensor[:, 1] * 1.57   # Shoulder joint
    Y_denorm[:, 2] = Y_tensor[:, 2] * 2.62   # Elbow joint
    Y_denorm[:, 3] = Y_tensor[:, 3] * 2.62   # Elbow2 joint
    Y_denorm[:, 4] = Y_tensor[:, 4] * 2.62   # Wrist1 joint

    # Sine/cosine representation
    Y_sin = torch.sin(Y_denorm)
    Y_cos = torch.cos(Y_denorm)
    Y_combined = torch.cat([Y_sin, Y_cos], dim=1)  # shape: [N, 10]

    print(f"Dataset shapes: X={X_tensor.shape}, Y_combined={Y_combined.shape}")

    # Dataset & split
    dataset = TensorDataset(X_tensor, Y_combined)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True, num_workers=4)

    # Define the network with skip connections
    class IKNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, 1024)
            self.fc5 = nn.Linear(1024, 512)
            self.out = nn.Linear(512, 10)
            self.skip = nn.Linear(3, 512)  # project input for skip connection
            self.act = nn.GELU()

        def forward(self, x):
            h = self.act(self.fc1(x))
            h = self.act(self.fc2(h))
            h = self.act(self.fc3(h))
            h = self.act(self.fc4(h))
            h = self.act(self.fc5(h))
            h = h + self.skip(x)  # skip connection
            return self.out(h)

    net = IKNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    num_epochs = 30

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize loss tracking
    loss_history = []

    # Training loop
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Evaluate on test set every epoch
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device, non_blocking=True), Y_batch.to(device, non_blocking=True)
                outputs = net(X_batch)
                test_loss += criterion(outputs, Y_batch).item() * X_batch.size(0)
        test_loss /= len(test_loader.dataset)

        # Store loss data
        loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss
        })

        # Print every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Save loss history to CSV
    with open('training_loss.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'test_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    print("📊 Loss history saved to 'training_loss.csv'")

    torch.save(net.state_dict(), "iknet_model.pth")
    print("✅ Model saved!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
