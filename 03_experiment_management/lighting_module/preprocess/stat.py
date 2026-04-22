import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import tqdm
        
# calculate image statistics
def get_mean_std(data_path, batch_size=32, num_workers=0):
    """calculate pixel mean and std

    Args:
        data_path (_type_): data path for dataset
        batch_size (int, optional): _description_. Defaults to 32.
        num_workers (int, optional): _description_. Defaults to 2.

    Returns:
        mean: tuple: mean for each channel
        std: tuple: std for each channel
    """
    
    # define a transform to convert PIL image to tensor
    transform = transforms.Compose([
        transforms.Resize((100, 100)),# we can compare, the results from 256 and 100 are pretty closed. #transforms.Resize((256, 256)),
        #transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # load dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # initialize mean and st
    channels_sum = torch.zeros(3)
    channels_sqrd_sum = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm.tqdm(loader):
        channels_sum += images.sum(dim=[0, 2, 3])
        channels_sqrd_sum += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.numel() / images.shape[1]  # total number of pixels per channel
        
    # calculate mean and std
    mean = channels_sum / num_pixels
    std = (channels_sqrd_sum / num_pixels - mean ** 2) ** 0.5

    return mean, std
