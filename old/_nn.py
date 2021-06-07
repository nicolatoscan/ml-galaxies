# %% imports
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomCrop, ToTensor, RandomVerticalFlip, RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from typing import Any, cast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
import torchvision.models as models
print("Done")


# %% Classification Metrics
class ClassificationMetrics:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.C = torch.zeros(num_classes, num_classes).to('cuda')
    def add(self, yp, yt):
        with torch.no_grad():
            self.C += (yt*self.C.shape[1]+yp).bincount(minlength=self.C.numel()).view(self.C.shape).float()
    def clear(self):
        self.C.zero_()
    def acc(self):
        return self.C.diag().sum().item()/self.C.sum()
    def mAcc(self):
        return (self.C.diag()/self.C.sum(-1)).mean().item()
    def mIoU(self):
        return (self.C.diag()/(self.C.sum(0)+self.C.sum(1)-self.C.diag())).mean().item()
    def confusion_matrix(self):
        return self.C

# %% NN train & validate
class TrainModel():
    name: str
    model: nn.Module
    trainLoader: DataLoader
    validateLoader: DataLoader
    mt: ClassificationMetrics
    tblog: SummaryWriter

    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler._LRScheduler
    lossFn: nn.Module
    epoch:int
    device: torch.device

    def __init__(   self,
                    name: str,
                    model: nn.Module,
                    trainLoader: DataLoader,
                    validateLoader: DataLoader,
                    optimizer: optim.Optimizer,
                    lr_scheduler: optim.lr_scheduler._LRScheduler,
                    lossFn: nn.Module
                ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.name = name
        self.model = model.to(self.device)
        self.trainLoader = trainLoader
        self.validateLoader = validateLoader
        self.mt = ClassificationMetrics(10)
        self.tblog = tb.SummaryWriter(f"exps/{name}")
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lossFn = lossFn
        self.epoch = 0

    def trainEpoch(self):
        self.epoch += 1
        self.model.train()
        self.mt.clear()
        for i, (X, yt) in enumerate(self.trainLoader):
            X = X.to(self.device)
            # X = X.to(self.device).reshape(X.size(0), -1)
            yt = yt.to(self.device)

            self.optimizer.zero_grad()
            scores = model(X)
            loss = self.lossFn(scores, yt)

            y = scores.argmax(-1)
            self.mt.add(y, yt)

            self.tblog.add_scalar('train/loss', loss.item(), self.epoch * len(self.trainLoader) + i)

            loss.backward()
            self.optimizer.step()

        self.tblog.add_scalar('train/acc', self.mt.acc(), self.epoch)
        self.tblog.add_scalar('train/mAcc', self.mt.mAcc(), self.epoch)
        self.tblog.add_scalar('train/mIoU', self.mt.mIoU(), self.epoch)
        self.tblog.add_scalar('train/lr', self.lr_scheduler.get_last_lr()[0], self.epoch)

    def validate(self):
        self.model.eval()
        self.mt.clear()
        with torch.no_grad():
            for X, yt in self.validateLoader:
                X = X.to('cuda')
                yt = yt.to('cuda')
                Y = model(X)
                # Y = model(X.reshape(X.size(0), -1))
                y = Y.argmax(-1)
                self.mt.add(y, yt)
        self.tblog.add_scalar('val/acc', self.mt.acc(), self.epoch)
        self.tblog.add_scalar('val/mAcc', self.mt.mAcc(), self.epoch)
        self.tblog.add_scalar('val/mIoU', self.mt.mIoU(), self.epoch)

    def train(self, nEpoch: int):
        for i in range(1, nEpoch):
            print(f"-- EPOCH {i}/{nEpoch} [Tot: {self.epoch}] -------------------------\n")
            self.trainEpoch()
            print(f"\tTRAIN | acc: {self.mt.acc():.4f} | mAcc: {self.mt.mAcc():.4f} | mIoU: {self.mt.mIoU():.4f}")
            self.validate()
            print(f"\tEVAL  | acc: {self.mt.acc():.4f} | mAcc: {self.mt.mAcc():.4f} | mIoU: {self.mt.mIoU():.4f}\n")
            self.lr_scheduler.step()


# %% load data
def load_dataset(path: str, trainset: bool = True) -> ImageFolder:
    transformations = []
    if trainset:
        transformations = [
            # ToPILImage(),
            RandomRotation(90),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transformations = [
            # ToPILImage(),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    transform = Compose(transformations)  # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    return dataset

def load_images(dataset: ImageFolder, batch_size:int=128, trainset: bool = True) -> DataLoader:
    sampler = None
    if trainset:
        classWeights = [1144, 187, 605, 1048, 796, 1135, 1037, 1480, 1472, 1024]
        sampleWeights = [0] * len(dataset)
        for idx, (_, label) in enumerate(cast(Any, dataset)):
            cw = classWeights[label]
            sampleWeights[idx] = cw
        sampler = WeightedRandomSampler(sampleWeights, num_samples=len(sampleWeights), replacement=True) if trainset else None
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True,
                        sampler=sampler)


datasetTrain = load_dataset('./dataset/train')
datasetValidate = load_dataset('./dataset/validate')
loaderTrain = load_images(datasetTrain, 128)
loaderValidate = load_images(datasetValidate, 128, trainset=False)
print("Done")

# %% ALex
model = models.resnet34(pretrained=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
trainingModel = TrainModel(
    name='pippo',
    model=model,
    trainLoader=loaderTrain,
    validateLoader=loaderValidate,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    lossFn=nn.CrossEntropyLoss()
)

trainingModel.train(5)

# %%
