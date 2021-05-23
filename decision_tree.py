# %% imports
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
loaderTrain = load_images(datasetTrain, 512)
loaderValidate = load_images(datasetValidate, 512)

# images, labels = next(iter(loaderTrain))
# images = cast(List[Tensor], images)
# plt.imshow(images[0]);

# %% decision tree
dtree = tree.DecisionTreeClassifier()
i = 0;
for images, labels in loaderTrain:
    print(i)
    i += 1
    images = cast(Tensor, images)
    images = images.mean(dim=3).reshape(images.size(0), -1)
    # plt.imshow(images[0].reshape(224,224), cmap='gray');
    dtree.fit(images, labels)

# %% prediction
gtLabels = torch.tensor([]).int()
pLabels = torch.tensor([]).int()
for images, labels in loaderValidate:
    images = cast(Tensor, images)
    images = images.mean(dim=3).reshape(images.size(0), -1)
    predictedLabels = torch.from_numpy(dtree.predict(images))
    print(labels)
    gtLabels = torch.cat((gtLabels, labels), 0)
    pLabels = torch.cat((pLabels, predictedLabels), 0)
x = evaluate(gtLabels, pLabels)
print(x)


# %%
