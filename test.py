# %% imports
from matplotlib import image
from scipy.sparse.dia import dia_matrix
from utils.evaluate import evaluate

from typing import cast

import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter, Normalize
from torchvision.datasets import ImageFolder

from sklearn import tree
import matplotlib.pyplot as plt
import concurrent.futures

# %% Load dataset
idx_to_class = {}
def load_dataset(path: str) -> ImageFolder:
    global idx_to_class
    transform = Compose([
        RandomCrop(224),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ColorJitter(0, 0, 0, 0),
        # lambda x: x.permute(1, 2, 0)
    ]) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return dataset

def load_images(dataset: ImageFolder, batch_size=128) -> DataLoader:
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True)

datasetTrain = load_dataset('./assets/dataset/train')
datasetValidate = load_dataset('./assets/dataset/validate')
loaderTrain = load_images(datasetTrain, 16)
loaderValidate = load_images(datasetValidate, 16)


# %%
Xs, labels = next(iter(loaderTrain))

# %%
for i, failure_idx in enumerate(Xs):
    ax=plt.subplot(4,4,i+1)
    ax.axis('off')
    plt.imshow(Xs[i].permute(1, 2, 0))
print(idx_to_class[labels[0].item()])

# %%
import math
X = torch.tensor([[0] * 224] * 224).float()
for i in range(0, 224):
    for j in range(0, 224):
        X[i][j] =  ((i - (224 / 2))**2) + ((j - (224 / 2))**2)
X /= X.max() * 3
X = torch.stack((X, X, X)).permute(1, 2, 0)
print(Xs.shape)
# X2s = torch.cat((
#     Xs,
#     Xs.rot90(1, dims=[2, 3]),
#     Xs.rot90(2, dims=[2, 3]),
#     Xs.rot90(3, dims=[2, 3]),
#     Xs.flip(dims=(2,)),
#     Xs.flip(dims=(3,)),
#     Xs.rot90(1, dims=[2, 3]).flip(dims=(2,)),
#     Xs.rot90(1, dims=[2, 3]).flip(dims=(3,)),
#     ))
X2s = Xs
print(X2s.shape)
for i, _ in enumerate(X2s):
    x = ColorJitter(0, 1, 0, 0)(Xs[i])
    ax=plt.subplot(len(X2s)/4,4,i+1)
    ax.axis('off')
    plt.imshow(x.permute(1, 2, 0))
print(idx_to_class[labels[0].item()])

# %%
