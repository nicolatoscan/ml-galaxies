# %% fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
import torch.utils.tensorboard as tb
from torchvision.transforms.transforms import Normalize


# %% my NN
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.num_classes = num_classes

        self.hidden1 = nn.Linear(input_dim, 500)
        self.hidden2 = nn.Linear(500, 500)
        self.hidden3 = nn.Linear(500, 100)
        self.output = nn.Linear(100, num_classes)
        self.activation = nn.Sigmoid()

        # for l in [self.hidden1, self.hidden2]:
        #     nn.init.xavier_normal_(l.weight)
        #   torch.nn.init.constant_(l.bias,0)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        return self.output(x)

# %% device
device = torch.device('cuda')
inputSize = 150528
nClass = 10
learningRate = 0.001
batchSize = 64
nEpochs = 1

# %% loaddata
# %% load data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor
from torchvision.datasets import ImageFolder

def load_dataset(path: str) -> ImageFolder:
    transformations = [
        RandomCrop(224),
        ToTensor(),
        Normalize(mean=[0.1758, 0.1712, 0.1670], std=[0.1381, 0.1279, 0.1205])
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
loaderTrain = load_images(datasetTrain, batchSize)
loaderValidate = load_images(datasetValidate, batchSize)

# %% init
model = SimpleMLP(inputSize, 10).to(device)

# %% loss and opt
critirion = nn.CrossEntropyLoss()
optiomizer = optim.Adam(model.parameters(), lr=learningRate)

# %% train
for epoch in range(nEpochs):
    for i, (data, y) in enumerate(loaderTrain):
        data = data.to(device).reshape(data.shape[0], -1)
        y = y.to(device)

        scores = model(data)
        loss = critirion(scores, y)

        optiomizer.zero_grad()
        loss.backward()
        optiomizer.step()
        if i % 10 == 0:
            print(i)



# %% check
def check(loader, model):
    nCorr = 0
    nSample = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).reshape(data.shape[0], -1)
            y = y.to(device)
            scores = model(x)
            _, pred = scores.max(1)
            nCorr = (pred == y).sum()
            nSample = pred.size(0)
        acc = nCorr / nSample
        print(acc)


check(loaderTrain, model)
check(loaderValidate, model)
# %%
