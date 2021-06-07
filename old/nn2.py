# %% imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
import torch.utils.tensorboard as tb

# %% fn
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

def trainEpoch(model, lossFn, metric_tracker, dataloader, optimizer, epoch, tblog=None):
    model.train()
    metric_tracker.clear()
    for i, (X, yt) in enumerate(dataloader):
        X = X.to('cuda')
        # X = X.to('cuda').reshape(X.size(0), -1)
        yt = yt.to('cuda')

        optimizer.zero_grad()
        scores = model(X)
        loss = lossFn(scores, yt)

        y = scores.argmax(-1)
        metric_tracker.add(y, yt)
        if tblog:
            tblog.add_scalar('train/loss', loss.item(), epoch*len(dataloader)+i)

        loss.backward()
        optimizer.step()

def validate(model, metric_tracker, dataloader):
    model.eval()
    metric_tracker.clear()
    with torch.no_grad():
        for X, yt in dataloader:
            X = X.to('cuda')
            yt = yt.to('cuda')
            Y = model(X)
            # Y = model(X.reshape(X.size(0), -1))
            y = Y.argmax(-1)
            metric_tracker.add(y, yt)

def train(model, trDataLoader, vlDataLoader, optimizer, lr_scheduler, num_epochs, tblog=None):
    lossFn = nn.CrossEntropyLoss()
    metric_tracker = ClassificationMetrics(10)

    for epoch in range(1, num_epochs+1):
        print(f"-- EPOCH {epoch}/{num_epochs} -------------------------\n")
        trainEpoch(model, lossFn, metric_tracker, trDataLoader, optimizer, epoch, tblog)
        print(f"\tTRAIN | acc: {metric_tracker.acc():.4f} | mAcc: {metric_tracker.mAcc():.4f} | mIoU: {metric_tracker.mIoU():.4f}")

        if tblog:
            tblog.add_scalar('train/acc', metric_tracker.acc(), epoch)
            tblog.add_scalar('train/mAcc', metric_tracker.mAcc(), epoch)
            tblog.add_scalar('train/mIoU', metric_tracker.mIoU(), epoch)
            tblog.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        validate(model, metric_tracker, vlDataLoader)

        print(f"\tEVAL  | acc: {metric_tracker.acc():.4f} | mAcc: {metric_tracker.mAcc():.4f} | mIoU: {metric_tracker.mIoU():.4f}\n")
        if tblog:
            tblog.add_scalar('val/acc', metric_tracker.acc(), epoch)
            tblog.add_scalar('val/mAcc', metric_tracker.mAcc(), epoch)
            tblog.add_scalar('val/mIoU', metric_tracker.mIoU(), epoch)
        lr_scheduler.step()

def run(name, model, trDataLoader, vlDataLoader):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)
    tblog = tb.SummaryWriter("exps/{}".format(name))
    train(model, trDataLoader, vlDataLoader, optimizer, lr_scheduler, num_epochs=25, tblog=tblog)

# %% load data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import ImageFolder

def load_dataset(path: str) -> ImageFolder:
    transformations = [
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomCrop(224),
        ToTensor(),
    ]
    transform = Compose(transformations) # remember to normalize the data
    dataset = ImageFolder(path, transform=transform)
    return dataset

def load_images(dataset: ImageFolder, batch_size=128, shuffle:bool=True) -> DataLoader:
    return DataLoader(  dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=2,
                        pin_memory=True,
                        drop_last=True)

datasetTrain = load_dataset('./assets/dataset/train')
datasetValidate = load_dataset('./assets/dataset/validate')
loaderTrain = load_images(datasetTrain, 2)
loaderValidate = load_images(datasetValidate, 2)

# %% NN
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, dims, num_classes):
        super(SimpleMLP, self).__init__()
        self.num_classes = num_classes
        self.hidden1 = nn.Linear(input_dim, dims[0])
        self.hidden2 = nn.Linear(dims[0], dims[1])
        self.output = nn.Linear(dims[1], num_classes)
        self.activation = nn.Sigmoid()

        for l in [self.hidden1, self.hidden2]:
            nn.init.xavier_normal_(l.weight)
        #   torch.nn.init.constant_(l.bias,0)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        return self.output(x)

class MyFullyConnectedNN(nn.Module):
    def __init__(self, num_channels, num_classes, activation_type=nn.ReLU):
        super(MyFullyConnectedNN, self).__init__()
        self.num_classes = num_classes
        activation = activation_type()
        layers = []
        for i in range(1,len(num_channels)):
            layers.append(nn.Linear(num_channels[i-1],num_channels[i]))
            layers.append(activation)
        layers.append(nn.Linear(num_channels[-1],num_classes))
        self.layers=nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
# # %% run
# merda = SimpleMLP(150528, [256, 128], 10).to('cuda')
# run('SimpleMLP', merda, loaderTrain, loaderValidate)

# # %% fullt
# modelFull=MyFullyConnectedNN(num_channels=[150528,256,128], num_classes=10, activation_type=nn.Sigmoid).to('cuda')
# run('MLP-256-128-ReLU', modelFull, loaderTrain, loaderValidate)


# %%
import torchvision.models as models
model = models.resnet34(pretrained=True)
numFeatures = model.fc.in_features
model.fc = nn.Linear(numFeatures, 10)

model = model.to('cuda')
run('RESNET50', model, loaderTrain, loaderValidate)

# %%
