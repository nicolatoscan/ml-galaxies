# %% import
from matplotlib import image
from utils.evaluate import evaluate

from typing import cast
import time

import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter, Normalize, Resize
from torchvision.datasets import ImageFolder

from sklearn import neighbors
import matplotlib.pyplot as plt
import concurrent.futures

# %% Load dataset
def load_dataset(path: str) -> ImageFolder:
    X = torch.tensor([[0] * 256] * 256).float()
    for i in range(0, 256):
        for j in range(0, 256):
            X[i][j] =  ((i - (256 / 2))**2) + ((j - (256 / 2))**2)
    X /= X.max() * 6
    X = torch.stack((X, X, X)).permute(1, 2, 0)

    transform = Compose([
        # RandomCrop(224),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ColorJitter(0, 0.5, 0, 0),
        # lambda x: x.permute(1, 2, 0),
        # lambda x: x - X,
        # Resize((128, 128))
    ]) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return dataset

def load_images(dataset: ImageFolder, batch_size=128) -> DataLoader:
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True)

datasetTrain = load_dataset('./assets/dataset/train')
datasetValidate = load_dataset('./assets/dataset/validate')
loaderTrain = load_images(datasetTrain, 256)
loaderValidate = load_images(datasetValidate, 256)


# %% traim
knn = neighbors.KNeighborsClassifier(n_jobs=-1)
start = time.time()
i = 0
for images, labels in loaderTrain:
    images = cast(Tensor, images)
    images = torch.cat((
        images,
        images.rot90(1, dims=[2, 3]),
        images.rot90(2, dims=[2, 3]),
        images.rot90(3, dims=[2, 3]),
        images.flip(dims=(2,)),
        images.flip(dims=(3,)),
        images.rot90(1, dims=[2, 3]).flip(dims=(2,)),
        images.rot90(1, dims=[2, 3]).flip(dims=(3,)),
    ))
    images = images.reshape(images.size(0), -1)
    labels = labels.repeat(8)
    knn.fit(images, labels)
    i += 1
    print(i)
end = time.time()
print(f"Training time: {end-start}s")

# %% Validate
gtLabels = torch.tensor([]).int()
pLabels = torch.tensor([]).int()

start = time.time()
i = 0
for images, labels in loaderValidate:
    images = cast(Tensor, images).reshape(images.size(0), -1)
    predictedLabels = torch.from_numpy(knn.predict(images))
    pLabels = torch.cat((pLabels, predictedLabels), 0)
    gtLabels = torch.cat((gtLabels, labels), 0)
    i += 1
    print(i)
end = time.time()
print(f"Validation time: {end-start}s")

# %% Evaluate
x = evaluate(gtLabels, pLabels)
print(x)
