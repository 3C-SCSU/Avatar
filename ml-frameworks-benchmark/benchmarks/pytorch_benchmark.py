"""
PyTorch Benchmark for EEG Classification
Measures performance of PyTorch on brainwave data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import BenchmarkDataLoader


class SimpleCNN(nn.Module):
    """Simple CNN for EEG classification"""
    def __init__(self, n_features=16, n_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (n_features // 4), 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input: (batch, features)
        x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, features)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """Train the PyTorch model"""
    model.train()
    training_times = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
    
    return training_times


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy and inference time"""
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return accuracy, avg_inference_time


def run_benchmark(device_type='cpu', epochs=10):
    """Run complete PyTorch benchmark"""
    print(f"\n{'='*60}")
    print(f"PyTorch Benchmark - Device: {device_type.upper()}")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(device_type if torch.cuda.is_available() or device_type == 'cpu' else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    loader = BenchmarkDataLoader(use_synthetic=False)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = SimpleCNN(n_features=16, n_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    total_train_start = time.time()
    training_times = train_model(model, train_loader, criterion, optimizer, device, epochs)
    total_train_time = time.time() - total_train_start
    
    # Evaluate
    print("\nEvaluating...")
    accuracy, avg_inference_time = evaluate_model(model, test_loader, device)
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Total Training Time: {total_train_time:.2f}s")
    print(f"Average Epoch Time: {sum(training_times)/len(training_times):.2f}s")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
    print(f"{'='*60}\n")
    
    return {
        'framework': 'PyTorch',
        'device': device_type,
        'total_train_time': total_train_time,
        'avg_epoch_time': sum(training_times)/len(training_times),
        'accuracy': accuracy,
        'avg_inference_time': avg_inference_time * 1000,  # Convert to ms
        'epochs': epochs
    }


if __name__ == "__main__":
    # Run CPU benchmark
    results_cpu = run_benchmark(device_type='cpu', epochs=5)
    
    # Try GPU if available
    if torch.cuda.is_available():
        results_gpu = run_benchmark(device_type='cuda', epochs=5)