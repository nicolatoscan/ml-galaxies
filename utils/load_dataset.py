from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter
from torchvision.datasets import ImageFolder
import torch


def load_dataset(path: str) -> ImageFolder:
    transform = Compose([
        RandomCrop(224),
        ToTensor(),
        ColorJitter(0, 0.5, 0, 0),
        lambda x: x.permute(1, 2, 0)
    ]) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return dataset

def load_images(dataset: ImageFolder, batch_size=128) -> DataLoader:
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)


