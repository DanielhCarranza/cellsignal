"""Base DataModule class."""
from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import argparse
from PIL import Image

import os

import torch
from torchvision import transforms as T
from torch.utils.data import  DataLoader
import pytorch_lightning as pl

DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
BATCH_SIZE = 128
NUM_WORKERS = 0

class RxRx1DataModule(pl.LightningDataModule):
    """Data module for RXRX1 dataset """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        self.split_size: Tuple[int,...]
        self.input_dims: Tuple[int,...]
        self.output_dims: Tuple[int,...]
    
    @property
    def base_path(self):
        return Path(__file__).resolve().parents[2] / "data/rxrx1/images"

    @staticmethod
    def load_image(filename:str):
        with Image.open(filename) as img:
            return T.ToTensor()(img)
    
    def load_images_as_tensor(self, image_paths):
        """Loads images from a list of paths and returns a [1, 6, 512, 512] tensor."""
        return torch.stack([self.load_image(path) for path in image_paths], dim=1)

    def image_path(self, experiment, plate, address, site, channel):
        return os.path.join(self.base_path, experiment, f"Plate{plate}", f"{address}_s{site}_w{channel}.png")
    
    def load_site(self, experiment, plate, address, site):
        """Loads a site from a given experiment, plate, address, and site."""
        image_paths = [self.image_path(experiment, plate, address, site, channel) for channel in DEFAULT_CHANNELS]
        return self.load_images_as_tensor(image_paths)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser
    
    def config(self):
        """Return important settings of the dataset"""
        return {"input_dims":self.input_dims, "output_dims":self.output_dims, "split_size": self.split_size}
    
    def prepare_data(self)->None:
        """Preprocessing to be done only from a single GPU"""
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Split data into train, val, test, and set dims"""
        self.split_size = (0.8, 0.1, 0.1)
        self.train_set, self.val_set, self.test_set = self.split_data()
    
    def train_dataloader(self):
        return DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=True, 
                pin_memory=self.on_gpu)
                
    def val_dataloader(self):
        return DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=True, 
                pin_memory=self.on_gpu)

    def test_dataloader(self):
        return DataLoader(
                self.test_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                shuffle=True, 
                pin_memory=self.on_gpu)
    
    def split_data(self)->Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Split data into train, val, test"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.train_set)
    



