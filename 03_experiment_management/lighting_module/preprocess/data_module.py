# data module template using PyTorch Lightning
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from preprocess import get_dataset, data_access

class FlowerDataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_transform, basic_transform, batch_size=32, num_workers=0):
        """initialize the data module
        Args:
            data_path: path to the dataset
            train_transform: transforms to apply to the training data
            basic_transform: transforms to apply to the validation and test data
            batch_size (int, optional): batch size. Defaults to 32.
            num_workers (int, optional): number of workers for data loading. Defaults to 0.
        """
        super().__init__()
        self.data_path = data_path
        self.train_transform = train_transform
        self.basic_transform = basic_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """setup the data module by applying transforms and creating datasets
        """
        # read data using predefined function
        dataset = data_access(self.data_path)
        # devide into train, val, test using predefined function
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
            dataset=dataset,
            train_transform=self.train_transform,
            basic_transform=self.basic_transform,
            train_size=0.80,
            val_size=0.15,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        """return training data loader
        """
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=True, pin_memory=True, prefetch_factor=2, persistent_workers=True, generator=g)

    def val_dataloader(self):
        """return validation data loader
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          persistent_workers=True, pin_memory=True )

    def test_dataloader(self):
        """return test data loader
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
