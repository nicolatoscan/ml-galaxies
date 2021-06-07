# %% imports
import time
from typing import Any, Dict, List, Tuple
import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import concurrent.futures
import pprint
import json
print("Done")

# %% Load dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale, RandomRotation, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import ImageFolder

def load_dataset(path: str, grayScale:bool, trainset:bool) -> ImageFolder:
    transformations: List = [ Grayscale(num_output_channels=1) ] if grayScale else []
    transformations += [ Resize(192 if grayScale  else 128) ]
    if trainset:
        transformations += [
            RandomRotation(90),
            RandomVerticalFlip(),
            RandomHorizontalFlip()
        ]
    transformations += [
        ToTensor(),
        Normalize(mean=[0.1639], std=[0.1068]) if grayScale else Normalize(mean=[0.1673, 0.1624, 0.1587], std=[0.1137, 0.1079, 0.1000])
    ]
    dataset = ImageFolder(path, transform=Compose(transformations))
    return dataset

def load_images(dataset: ImageFolder, batch_size=128, trainset:bool=True, sampleWeights:List[float] = []) -> DataLoader:
    sampler = WeightedRandomSampler(sampleWeights, num_samples=len(sampleWeights), replacement=True) if trainset else None
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=False,
                        sampler=sampler)

datasetTrainGrayScale = load_dataset('./dataset/train', True, True)
datasetTrain = load_dataset('./dataset/train', False, True)
datasetValidateGrayScale = load_dataset('./dataset/validate', True, False)
datasetValidate = load_dataset('./dataset/validate', False, False)

# %% sampler
def getSampleWeights(dataset: ImageFolder) -> List[float]:
    class_counts = [1144, 187, 605, 1048, 796, 1135, 1037, 1480, 1472, 1024]
    num_samples = sum(class_counts)
    labels = []
    for _, label in tqdm(dataset):
        labels.append(label)
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    return weights
sampleWeights = getSampleWeights(datasetTrain)

# %% loader
loaderTrain = load_images(datasetTrain, 128, trainset=True, sampleWeights=sampleWeights)
loaderValidate = load_images(datasetValidate, 128, trainset=False)
loaderTrainGrayScale = load_images(datasetTrainGrayScale, 128, trainset=True, sampleWeights=sampleWeights)
loaderValidateGrayScale = load_images(datasetValidateGrayScale, 128, trainset=False)
print("Done")


# %% load
def parseImageLabels(imageLables: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    return (imageLables[0].reshape(imageLables[0].size(0), -1), imageLables[1])

images = torch.Tensor([])
labels = torch.Tensor([]).int()
for i, l in tqdm(loaderTrain):
    i , l = parseImageLabels((i, l))
    images = torch.cat((images, i), 0)
    labels = torch.cat((labels, l), 0)

imagesVal = torch.Tensor([])
labelsVal = torch.Tensor([]).int()
for i, l in tqdm(loaderValidate):
    i , l = parseImageLabels((i, l))
    imagesVal = torch.cat((imagesVal, i), 0)
    labelsVal = torch.cat((labelsVal, l), 0)

print("Done")

# %% functions
times: Dict[str, Dict[str, float]] = { "train": { }, "predict": { }}
res: Dict[str, Dict[str, float]] = {}

def fitModel(model, images: Tensor, labels: Tensor, name: str):
    start = time.time()
    model.fit(images, labels)
    end = time.time()
    times["train"][name] = end - start
    print(f"Done {name}")

def train(models: List[Tuple[Any, str]], images: Tensor, labels: Tensor, max_workers: int = 4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(lambda m: fitModel(m[0], images, labels, m[1]), models))

def predictModel(model, name: str, pLabels: Dict[str, Tensor]):
    start = time.time()
    Y = model.predict(imagesVal)
    end = time.time()
    times["predict"][name] = end - start

    pLabels[name] = torch.from_numpy(Y)
    print(f"Done {name}")


def predict(models: List[Tuple[Any, str]], max_workers: int = 2) -> Dict[str, Tensor]:
    pLabels: Dict[str, Tensor] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            executor.map(lambda m: predictModel(m[0], m[1], pLabels)
            , models)
        )
    return pLabels

def evaluate(gtLabels: torch.Tensor, pLabels: torch.Tensor, num_classes=10) -> Dict[str, float]:
    C=(gtLabels*num_classes+pLabels).bincount(minlength=num_classes**2).view(num_classes,num_classes).float()
    return {
        'Acc': C.diag().sum().item() / gtLabels.shape[0],
        'mAcc': (C.diag()/C.sum(-1)).mean().item(),
        'mIoU': (C.diag()/(C.sum(0)+C.sum(1)-C.diag())).mean().item()
    }
def evaluateAll(gtLabels: Tensor, pLabels: Dict[str, Tensor]) -> Dict[str, Dict[str, float]]:
    for k in pLabels:
        res[k] = evaluate(gtLabels, pLabels[k])
    return res

def printEvaluation(res: Dict[str, Dict[str, float]]):
    for k in res:
        print(f"\n{k:15} Acc: {res[k]['Acc']:10} mAcc: {res[k]['mAcc']:10} mIoU: {res[k]['mIoU']:10}")

def resultOnModels(models: List, fileName: str):
    pLabels = predict(models)
    res = evaluateAll(labelsVal, pLabels)
    pprint.pprint(res)
    with open(fileName, "w") as f:
        json.dump({ "res": res, "times": times }, f)

print("Done")

# %% Models
from sklearn import tree, ensemble, neighbors, svm

# %% Decision Tree
models = [
    (tree.DecisionTreeClassifier(min_samples_leaf=nr, criterion=criterion), f'dTree {criterion}{nr}')
    for criterion in ["gini", "entropy"]
    for nr in [1, 5, 10, 20]
]
train(models, images=images, labels=labels)
resultOnModels(models, "dtree.json")
print("dTree")

# %% Forest
models = [
    (ensemble.RandomForestClassifier(n_estimators=nr, criterion=criterion), f'Forest {criterion}{nr}')
    for criterion in ["gini", "entropy"]
    for nr in [1, 2, 5, 10, 20, 50, 100, 200]
]
train(models, images=images, labels=labels)
resultOnModels(models, "Forest.json")
print("Forest")

# %% KNN
models = [
    (neighbors.KNeighborsClassifier(n_jobs=-1, n_neighbors=k), f'KNN {k}')
    for k in [1, 2, 5, 7, 9]
]
for model in models:
    train([model], images=images, labels=labels)
    resultOnModels([model], "KNN.json")
print("KNN")

# %% SVM
models = [
    (svm.SVC(kernel=kernel, C=C), f'SVM {kernel}{C}')
    for kernel in ['rbf', 'linear', 'poly'] 
    for C in [0.1, 1]
]
for model in models:
    train([model], images=images, labels=labels)
    resultOnModels([model], "SVM.json")
print("SVM")