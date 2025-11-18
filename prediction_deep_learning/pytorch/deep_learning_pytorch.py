import torch
import torch.nn as nn
import torch.nn.functional as F

# Create model class
class FlexibleCNNClassifier(nn.Module):  # Defines a custom class, must subclass nn.Module to access functions 
    def __init__(self, num_classes=6):
        super(FlexibleCNNClassifier, self).__init__() # used for initialization, required for every PyTorch model

        # Two 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)   # max pooling to control overfitting
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # shrinks sequence length to 1, needed before feeding to fully connected (Linear) layers

        # Two Linear layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes) # map to the # of output classes, which is 6 

    def forward(self, x): # Defines how data will be passed through the layers, using ReLU (introduces nonlinearity, which is important for learning complex patterns)
        if x.dim() == 2:  
            x = x.unsqueeze(1)  

        x = F.relu(self.conv1(x))
        x = self.pool(x) if x.shape[-1] > 1 else x
        x = F.relu(self.conv2(x))
        x = self.pool(x) if x.shape[-1] > 1 else x

        x = self.global_pool(x)       
        x = torch.flatten(x, 1)       
        x = F.relu(self.fc1(x))       
        x = self.fc2(x)              
        return x