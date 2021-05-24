# %% imports
from typing import Any, Dict, List, Tuple

import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter
from torchvision.datasets import ImageFolder

from sklearn import tree
import matplotlib.pyplot as plt
import tqdm
import concurrent.futures


# %% Functions
def load_dataset(path: str) -> ImageFolder:
    transform = Compose([
        RandomCrop(224),
        ToTensor(),
        # ColorJitter(0, 0.5, 0, 0),
        lambda x: x.permute(1, 2, 0)
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

def parseImageLabels(images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    return (images.reshape(images.size(0), -1), labels)

def trian(models: List[Tuple[Any, str]]):
    for i, l in tqdm(loaderTrain):
        (images, labels) = parseImageLabels(i, l)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(lambda m: m[0].fit(images, labels), models)
            )
        break
    print("Done")

def predictModel(model, images: Tensor, name: str, pLabels: Dict[str, Tensor]):
    predictedLabels = torch.from_numpy(model.predict(images))
    pLabels[name] = torch.cat((pLabels[name], predictedLabels), 0)

def predict(models: List[Tuple[Any, str]]) -> Tuple[Tensor, Dict[str, Tensor]]:
    gtLabels = torch.tensor([]).int()
    pLabels: Dict[str, Tensor] = {}
    for m, name in models:
        pLabels[name] = torch.tensor([]).int()
    for i, l in tqdm(loaderValidate):
        (images, labels) = parseImageLabels(i, l)
        gtLabels = torch.cat((gtLabels, labels), 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(lambda m: predictModel(m[0], images, m[1], pLabels)
                , models)
            )
    return (gtLabels, pLabels)

def evaluate(yt: torch.Tensor, yp: torch.Tensor, num_classes=10) -> Dict[str, float]:
    C=(yt*num_classes+yp).bincount(minlength=num_classes**2).view(num_classes,num_classes).float()
    return {
        'Acc': C.diag().sum().item() / yt.shape[0],
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


# %% train


# %%
