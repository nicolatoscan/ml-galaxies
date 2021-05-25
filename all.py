# %% imports
from typing import Any, Dict, List, Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures


# %% Functions
def load_dataset(path: str, isTrain:bool=True) -> ImageFolder:
    transformations = []
    if isTrain:
        transformations = [
            RandomCrop(224),
            ToTensor(),
            # ColorJitter(0, 0.5, 0, 0),
            # lambda x: x.permute(1, 2, 0)
        ]
    else:
        transformations = [
            ToTensor(),
            # lambda x: x.permute(1, 2, 0)
        ]

    transform = Compose(transformations) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return dataset

def load_images(dataset: ImageFolder, batch_size=128, shuffle:bool=True) -> DataLoader:
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True)

def parseImageLabels(images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    return (images.reshape(images.size(0), -1), labels)

def train(models: List[Tuple[Any, str]], max_workers: int = 8):
    for i, l in tqdm(loaderTrain):
        (images, labels) = parseImageLabels(i, l)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                executor.map(lambda m: m[0].fit(images, labels), models)
            )
        break
    print("Done")

def predictModel(model, images: Tensor, name: str, pLabels: Dict[str, Tensor]):
    predictedLabels = torch.from_numpy(model.predict(images))
    pLabels[name] = torch.cat((pLabels[name], predictedLabels), 0)

def predict(models: List[Tuple[Any, str]], max_workers: int = 4) -> Tuple[Tensor, Dict[str, Tensor]]:
    gtLabels = torch.tensor([]).int()
    pLabels: Dict[str, Tensor] = {}
    for _, name in models:
        pLabels[name] = torch.tensor([]).int()
    for i, l in tqdm(loaderValidate):
        (images, labels) = parseImageLabels(i, l)
        gtLabels = torch.cat((gtLabels, labels), 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(
                executor.map(lambda m: predictModel(m[0], images, m[1], pLabels)
                , models)
            )
        break
    return (gtLabels, pLabels)

def evaluate(gtLabels: torch.Tensor, pLabels: torch.Tensor, num_classes=10) -> Dict[str, float]:
    C=(gtLabels*num_classes+pLabels).bincount(minlength=num_classes**2).view(num_classes,num_classes).float()
    return {
        'Acc': C.diag().sum().item() / gtLabels.shape[0],
        'mAcc': (C.diag()/C.sum(-1)).mean().item(),
        'mIoU': (C.diag()/(C.sum(0)+C.sum(1)-C.diag())).mean().item()
    }
def evaluateAll(gtLabels: Tensor, pLabels: Dict[str, Tensor]) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for k in pLabels:
        res[k] = evaluate(gtLabels, pLabels[k])
    return res

# %% Load dataset
datasetTrain = load_dataset('./assets/dataset/train')
datasetValidate = load_dataset('./assets/dataset/validate')
loaderTrain = load_images(datasetTrain, 256)
loaderValidate = load_images(datasetValidate, 256)

# %% Create models
from sklearn import tree, ensemble, neighbors, svm

models: List[Tuple[Any, str]] = []
# Decision Tree
models += [
    (tree.DecisionTreeClassifier(min_samples_leaf=1, criterion='gini'), f'dTree {criterion}{nr}')
    for nr in [1, 5]
    for criterion in ["gini", "entropy"]
]
# Forest
models += [
    (ensemble.RandomForestClassifier(n_estimators=1, criterion='gini'), f'Forest {criterion}{nr}')
    for nr in [1, 5, 10, 20, 50, 100]
    for criterion in ["gini", "entropy"]
]
# KNN
models += [
    (neighbors.KNeighborsClassifier(n_jobs=-1), 'KNN'),
]
# SVM
models += [
    (svm.SVC(kernel=kernel, C=C), f'SVM {kernel}{C}')
    for kernel in ['rbf', 'linear', 'poly'] 
    for C in [0.1, 1, 10]
]
print("Model created")

# %% Train
print("Training ...")
train(models)
print("Trained")

# %% Predict
print("Predicting ...")
gtLabels, pLabels = predict(models)
res = evaluateAll(gtLabels, pLabels)
print("Predicted")

