# %% imports
from matplotlib import image
from utils.evaluate import evaluate

from typing import cast

import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ColorJitter
from torchvision.datasets import ImageFolder

from sklearn import tree
import matplotlib.pyplot as plt
import concurrent.futures

# %% Load dataset
def load_dataset(path: str) -> ImageFolder:
    transform = Compose([
        RandomCrop(224),
        ToTensor(),
        ColorJitter(0, 0.5, 0, 0),
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

datasetTrain = load_dataset('./assets/dataset/train')
datasetValidate = load_dataset('./assets/dataset/validate')
loaderTrain = load_images(datasetTrain, 256)
loaderValidate = load_images(datasetValidate, 256)

# images, labels = next(iter(loaderTrain))
# images = cast(List[Tensor], images)
# plt.imshow(images[0]);

# %% decision tree
dtrees = [
    (tree.DecisionTreeClassifier(min_samples_leaf=ms, criterion=criterion), ms, criterion)
    for ms in [1,5]
    for criterion in ["gini", "entropy"]
]
i = 0;
for images, labels in loaderTrain:
    print(i)
    i += 1
    images = cast(Tensor, images)
    images = images.reshape(images.size(0), -1)
    # plt.imshow(images[0].reshape(224,224), cmap='gray');
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        iterator = executor.map(lambda t: t[0].fit(images, labels), dtrees)
        list(iterator)
    # for t in dtrees:
    #     t[0].fit(images, labels)
print("Done")
# %% prediction
gtLabels = torch.tensor([]).int()
pLabels = {
    "gini": { 1: torch.tensor([]).int(), 5: torch.tensor([]).int() },
    "entropy": { 1: torch.tensor([]).int(), 5: torch.tensor([]).int() }
}
i = 0
for images, labels in loaderValidate:
    print(i)
    i += 1
    images = cast(Tensor, images)
    images = images.reshape(images.size(0), -1)
    for tree, ms, crit in dtrees:
        predictedLabels = torch.from_numpy(tree.predict(images))
        pLabels[crit][ms] = torch.cat((pLabels[crit][ms], predictedLabels), 0)
    gtLabels = torch.cat((gtLabels, labels), 0)


# %% show evaluations
for crit in pLabels:
    for ms in pLabels[crit]:
        x = evaluate(gtLabels, pLabels[crit][ms])
        print(crit, ms)
        print(x)


# %%
