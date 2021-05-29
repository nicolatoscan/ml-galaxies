# %% import
from tqdm import tqdm
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

from sklearn import ensemble
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
        lambda x: x.permute(1, 2, 0),
        lambda x: x - X,
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

# %% Train
randomForests = [
    (ensemble.RandomForestClassifier(n_estimators=5, criterion=criterion), nr, criterion)
    for nr in [1, 10, 20, 50, 100]
    for criterion in ["gini", "entropy"]
]
start = time.time()
i = 0
for images, labels in tqdm(loaderTrain):
    print(i)
    i += 1
    images = cast(Tensor, images)
    images = torch.cat((
            images,
            # images.rot90(1, dims=[2, 3]),
            # images.rot90(2, dims=[2, 3]),
            # images.rot90(3, dims=[2, 3]),
            # images.flip(dims=(2,)),
            # images.flip(dims=(3,)),
            # images.rot90(1, dims=[2, 3]).flip(dims=(2,)),
            # images.rot90(1, dims=[2, 3]).flip(dims=(3,)),
        ))
    labels = labels.repeat(1)
    images = images.reshape(images.size(0), -1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        iterator = executor.map(lambda t: t[0].fit(images, labels), randomForests)
        list(iterator)
end = time.time()
print(f"Training time: {end-start}s")

# %% Validate
gtLabels = torch.tensor([]).int()
pLabels = { }
for crit in ["gini", "entropy"]:
    pLabels[crit] = {}
    for nr in [1, 10, 20, 50, 100]:
        pLabels[crit][nr] = torch.tensor([]).int()
i = 0
start = time.time()
for images, labels in loaderValidate:
    print(i)
    i += 1
    images = cast(Tensor, images)
    images = images.reshape(images.size(0), -1)
    for tree, ms, crit in randomForests:
        predictedLabels = torch.from_numpy(tree.predict(images))
        pLabels[crit][ms] = torch.cat((pLabels[crit][ms], predictedLabels), 0)
    gtLabels = torch.cat((gtLabels, labels), 0)
end = time.time()
print(f"Evaluation time: {end-start}s")

# %% Evaluate
i = 0
acc = 0
for crit in pLabels:
    for ms in pLabels[crit]:
        i += 1
        x = evaluate(gtLabels, pLabels[crit][ms])
        print(f"{crit}\t{ms}\t{x['Acc']}")
        acc += x['Acc']
        # print(x['mAcc'])
        # print(x['mIoU'])
print(acc / i)
# %%


[1144, 187, 605, 1048, 796, 1135, 1037, 1480, 1472, 1024]