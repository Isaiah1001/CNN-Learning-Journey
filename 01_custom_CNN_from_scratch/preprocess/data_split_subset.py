import torch
import numpy as np
from .subset_class import subsetTrans
from torch.utils.data import  DataLoader, random_split, Subset

# get dataloaders
def get_dataloaders(dataset, batch_size, train_transform, val_transform, train_size=0.7, val_size=0.15):
    """create dataloaders for training, validation, and testing

    Args:
        dataset (data_access): dataset object
        batch_size (int): batch size for dataloaders
        train_transform (torchvision.transforms.Compose): transform for training data
        val_transform (torchvision.transforms.Compose): transform for validation data
        train_size (float, optional): proportion of training data. Defaults to 0.7.
        val_size (float, optional): proportion of validation data. Defaults to 0.15.
    Returns:
        train_loader, val_loader, test_loader: dataloaders for training, validation, and testing
    """

    # calculate sizes
    total_size = len(dataset)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size
    # split dataset
    np.random.seed(42)  # Set a fixed random seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # apply transforms
    train_dataset = subsetTrans(train_dataset, transform=train_transform)
    val_dataset = subsetTrans(val_dataset, transform=val_transform)
    test_dataset = subsetTrans(test_dataset, transform=val_transform)
    # create dataloaders
    torch.Generator().manual_seed(42) # set seed for dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
