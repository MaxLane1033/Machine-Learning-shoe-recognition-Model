"""
CNN model for shoe brand classification (Nike vs Adidas)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShoeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ShoeClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming input size 224x224
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def create_model(num_classes=2):
    """Create and return the shoe classifier model"""
    model = ShoeClassifier(num_classes=num_classes)
    return model

if __name__ == "__main__":
    # Test the model
    model = create_model()
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
