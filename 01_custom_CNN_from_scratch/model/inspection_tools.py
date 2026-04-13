#model/inspection_tools.py
from .model_architecture import SimpleCNN
import torch

class model_detail(SimpleCNN):
    """check model
    """
    def __init__(self):
        super().__init__() # initialize parent class
        
    def forward(self, x):
        print(f'Input shape: {x.shape}')
        print(f'Layer 1 (CNNBlock) output shape: {self.conv1.block(x).shape}')
        x = self.conv1.block(x)
        print(f'Layer 2 (CNNBlock) output shape: {self.conv2.block(x).shape}')
        x = self.conv2.block(x)
        print(f'Layer 3 (CNNBlock) output shape: {self.conv3.block(x).shape}')
        x = self.conv3.block(x)
        print(f'Layer 4 (CNNBlock) output shape: {self.conv4.block(x).shape}')
        x = self.conv4.block(x)
        print(f'Classifier input shape: {x.shape}')
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            print(f"classifier[{i}] {layer.__class__.__name__} -> {x.shape}")
        return x

class para_debug(SimpleCNN):
    def __init__(self):
        super().__init__()
        # The super().__init__() call above properly initializes all layers from SimpleCNN
        # No need to redefine the layers here

    def get_statistics(self, activation):
        mean = activation.mean().item()
        std = activation.std().item()
        min_val = activation.min().item()
        max_val = activation.max().item()

        print(f" Mean: {mean}")
        print(f" Std: {std}")
        print(f" Min: {min_val}")
        print(f" Max: {max_val}")
        return mean, std, min_val, max_val

    def forward(self, x):
        # block 1
        features = self.conv1(x)
        x1 = torch.flatten(features, start_dim=1)  # Flatten all dimensions except batch

        print("After conv block 1, the activation statistics are:")
        self.get_statistics(features)
        # block 2
        features = self.conv2(features)
        x2 = torch.flatten(features, start_dim=1)  # Flatten all dimensions except batch

        print("After conv block 2, the activation statistics are:")
        self.get_statistics(features)
        # block 3
        features = self.conv3(features)
        x = torch.flatten(features, start_dim=1)  # Flatten all dimensions except batch

        print("After conv block 3, the activation statistics are:")
        self.get_statistics(features)
        # block 4
        features = self.conv4(features)
        x = torch.flatten(features, start_dim=1)  # Flatten all dimensions except batch

        print("After conv block 4, the activation statistics are:")
        self.get_statistics(features)
        # fully connected block
        x = self.classifier(features)
        print("After classifier, the activation statistics are:")
        self.get_statistics(x)
        return x

def check_layer_parameters(model_input):
    """check the model layers briefly

    Args:
        model_input: model to be checked
    """
    # Iterate through the main blocks
    for name, block in model_input.named_children():
        print(f"Block {name} has a total of {len(list(block.children()))} layers:")
        # List all children layers in the block
        for idx, layer in enumerate(block.children()):
            # Check if the layer is terminal (no children) or not
            if len(list(layer.children())) == 0:
                print(f"\t {idx} - Layer {layer}")
            # If the layer has children, it's a sub-block, then print only the number of children and its name
            else:
                layer_name = layer._get_name()  # More user-friendly name
                print(f"\t {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")
            
                
def check_total_parameters(model_input):
    """print total number of parameter

    Args:
        model_input: model to be checked
    """
    total_params = sum(p.numel() for p in model_input.parameters())
    print(f"Total number of parameters in the model: {total_params}")