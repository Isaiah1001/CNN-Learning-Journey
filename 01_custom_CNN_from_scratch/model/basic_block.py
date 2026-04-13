#model/basic_block.py
import torch.nn as nn

class CNNBlock(nn.Module):
    """CNN block for building advanced model
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1, stride=1):
        """Initialization

        Args:
            in_channels (_type_): input channels
            out_channels (_type_): output channels
            kernel_size (int, optional): kernel size. Defaults to 3.
            padding (int, optional): padding. Defaults to 1.
            stride (int, optional): stride. Defaults to 1.
        """
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(num_features=out_channels), # batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def forward(self, x):
        return self.block(x)
    