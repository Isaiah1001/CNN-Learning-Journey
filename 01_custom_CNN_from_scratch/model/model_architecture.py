#model/model_architecture.py
import torch.nn as nn
from .basic_block import CNNBlock

class SimpleCNN(nn.Module):
    """Simple CNN model
    """
    def __init__(self, in_channels=3, num_classes=102):
        """Initialization

        Args:
            in_channels (int, optional): input channels. Defaults to 3.
            num_classes (int, optional): number of classes. Defaults to 102.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = CNNBlock(in_channels, 16)
        self.conv2 = CNNBlock(16, 32)
        self.conv3 = CNNBlock(32, 64)
        self.conv4 = CNNBlock(64, 128)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*14*14, 256), # assuming input image size is 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x
    
