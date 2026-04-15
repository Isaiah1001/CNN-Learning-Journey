import torch
import torchvision.models as tv_models
def build_efficientnet_b0(num_classes=102, weights=None):
    """build efficientnet_b0 model with modified head"""
    model = tv_models.efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    return model

def unfreeze_last_block_and_head(model, num_blocks=1):
    """Unfreeze the last N blocks and the classifier head of the model"""
    for param in model.parameters():
        param.requires_grad = False

    for i in range(1, num_blocks + 1):
        for param in model.features[-i].parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"Trainable modules: features[-{num_blocks}:] and classifier")

