# %% Imports
import os
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
import torch.nn.functional as fn
import torch.utils.tensorboard as tb
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
print("Done")

# %% Metrics
class ClassificationMetrics:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.C = torch.zeros(num_classes, num_classes).to('cuda')
    def add(self, yp, yt):
        with torch.no_grad():  # We require no computation graph
            self.C += (yt*self.C.shape[1]+yp).bincount(
                minlength=self.C.numel()).view(self.C.shape).float()
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
print("Done")

# %% Train model
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
            print("A")
            scores = self.model(X)
            print("B")
            loss = self.lossFn(scores, yt)

            y = scores.argmax(-1)
            self.mt.add(y, yt)

            print("C")
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
                Y = self.model(X)
                # Y = model(X.reshape(X.size(0), -1))
                y = Y.argmax(-1)
                self.mt.add(y, yt)
        self.tblog.add_scalar('val/acc', self.mt.acc(), self.epoch)
        self.tblog.add_scalar('val/mAcc', self.mt.mAcc(), self.epoch)
        self.tblog.add_scalar('val/mIoU', self.mt.mIoU(), self.epoch)

    def train(self, nEpoch: int):
        for i in range(0, nEpoch):
            print(f"-- EPOCH {i + 1}/{nEpoch} [Tot: {self.epoch + 1}] -------------------------\n")
            self.trainEpoch()
            print(f"\tTRAIN | acc: {self.mt.acc():.4f} | mAcc: {self.mt.mAcc():.4f} | mIoU: {self.mt.mIoU():.4f}")
            self.validate()
            print(f"\tEVAL  | acc: {self.mt.acc():.4f} | mAcc: {self.mt.mAcc():.4f} | mIoU: {self.mt.mIoU():.4f}\n")
            self.lr_scheduler.step()

    def getFeaures(self, dl: DataLoader):
        features = []
        targets=[]
        with torch.no_grad():
            for X,yt in dl:
                features.append(self.model(X, return_features=True))
                targets.append(yt)

print("Done")

# %% Load data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose, RandomCrop, ToTensor, RandomVerticalFlip, RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation
from torchvision.datasets import ImageFolder

def load_dataset(path: str, trainset:bool=True) -> ImageFolder:
    transformations = []
    if trainset:
        transformations = [
            RandomRotation(90),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        transformations = [
            RandomCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    transform = Compose(transformations) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    return dataset

def load_images(dataset: ImageFolder, batch_size=128, trainset:bool=True) -> DataLoader:
    sampler = None
    if trainset:
        classWeights = [1144, 187, 605, 1048, 796, 1135, 1037, 1480, 1472, 1024]
        sampleWeights = [0] * len(dataset)
        for idx, (_, label) in enumerate(tqdm(dataset)):
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
loaderTrain = load_images(datasetTrain, 64)
loaderValidate = load_images(datasetValidate, 64, trainset=False)
print("Done")


# %% My ConvNet
class ConvNet(nn.Module):
    def __init__(self, channels = 3, nClass = 10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(64*16*56*56, nClass)

        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(fn.relu(self.conv1(x)))
        x = self.pool(fn.relu(self.conv2(x)))
        print(x.shape)
        # x = self.pool(fn.relu(self.conv2(x)))
        # x = x.view(-1, 16*106*106)
        # x = fn.relu(self.fc1(x))
        # x = fn.relu(self.fc2(x))
        x = self.fc1(x)
        return x
# %%
model = ConvNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

trainingModel = TrainModel(
    name='CACCA',
    model=model,
    trainLoader=loaderTrain,
    validateLoader=loaderValidate,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    lossFn=nn.CrossEntropyLoss()
)

trainingModel.train(10)

# %%
import matplotlib.pyplot as plt
conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)

for x, _ in loaderTrain:
    img = x[0]
    plt.imshow(img.permute(1, 2, 0))

    x = conv1(x)
    x = pool(x)
    x = conv2(x)
    print(x.shape)

    break
# %%
