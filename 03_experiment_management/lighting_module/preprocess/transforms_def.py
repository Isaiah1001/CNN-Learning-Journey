import torchvision.transforms as transforms
# data manipulate
def data_manipulate(mean, std):
    """apply data manipulation to the image

    Args:
        mean (tuple): mean for each channel
        std (tuple): std for each channel
    Returns:
        basic (torchvision.transforms.Compose): basic transform
        aug (torchvision.transforms.Compose): basic + augmented transform
    """
    transform_basic = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    

    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10)
    ])
    # augmentation for training data and basic transform for validation/test data
    basic = transforms.Compose([transform_basic])
    aug = transforms.Compose([transform_basic, transform_aug])
    return basic, aug