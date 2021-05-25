# %% fn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb


class ClassificationMetrics:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.C = torch.zeros(num_classes, num_classes)
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


def train_one_epoch(model, loss_func, metric_tracker, dataloader, optimizer, epoch, tblog=None):
    model.train()
    metric_tracker.clear()
    for i, (X, yt) in enumerate(dataloader):
        X = X.reshape(X.size(0), -1).to('cuda')
        yt = yt.to('cuda')
        optimizer.zero_grad()
        Y = model(X)
        loss = loss_func(Y, yt)
        y = Y.argmax(-1)
        metric_tracker.add(y, yt)
        if tblog:
            tblog.add_scalar('train/loss', loss.item(),
                                epoch*len(dataloader)+i)
        loss.backward()
        optimizer.step()


def validate(model, metric_tracker, dataloader):
    model.eval()
    metric_tracker.clear()
    with torch.no_grad():
        for i, (X, yt) in enumerate(dataloader):
            Y = model(X)
            y = Y.argmax(-1)
            metric_tracker.add(y, yt)


def train(model, trDataLoader, vlDataLoader, optimizer, lr_scheduler, num_epochs, tblog=None):
    loss_func = nn.CrossEntropyLoss()
    metric_tracker = ClassificationMetrics(model.num_classes)

    for epoch in range(1, num_epochs+1):
        print("-- EPOCH {}/{} -------------------------\n".format(epoch, num_epochs))
        train_one_epoch(model, loss_func, metric_tracker,
                        trDataLoader, optimizer, epoch, tblog)
        print("\tTRAIN | acc: {:.4f} | mAcc: {:.4f} | mIoU: {:.4f}".format(
            metric_tracker.acc(), metric_tracker.mAcc(), metric_tracker.mIoU()
        ))
        if tblog:
            tblog.add_scalar('train/acc', metric_tracker.acc(), epoch)
            tblog.add_scalar('train/mAcc', metric_tracker.mAcc(), epoch)
            tblog.add_scalar('train/mIoU', metric_tracker.mIoU(), epoch)
            tblog.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)
        validate(model, metric_tracker, vlDataLoader)
        print("\tEVAL  | acc: {:.4f} | mAcc: {:.4f} | mIoU: {:.4f}\n".format(
            metric_tracker.acc(),
            metric_tracker.mAcc(), metric_tracker.mIoU()
        ))
        if tblog:
            tblog.add_scalar('val/acc', metric_tracker.acc(), epoch)
            tblog.add_scalar('val/mAcc', metric_tracker.mAcc(), epoch)
            tblog.add_scalar('val/mIoU', metric_tracker.mIoU(), epoch)
        lr_scheduler.step()


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.num_classes = num_classes
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, num_classes)
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

def run_experiment(name, model, trDataLoader, vlDataLoader):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.000001)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.5)
    tblog = tb.SummaryWriter("exps/{}".format(name))
    train(model, trDataLoader, vlDataLoader, optimizer, lr_scheduler, num_epochs=25, tblog=tblog)


# %% load data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor
from torchvision.datasets import ImageFolder

def load_dataset(path: str) -> ImageFolder:
    transformations = [
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
loaderTrain = load_images(datasetTrain, 256)
loaderValidate = load_images(datasetValidate, 256)

# %% run
model = SimpleMLP(150528, 256, 128, 10)
run_experiment('SimpleMLP', model, loaderTrain, loaderValidate)
# %%
