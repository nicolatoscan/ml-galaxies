# %% import
import json
from typing import List
import matplotlib.pyplot as plt

# %% load file
with open('tuned-resnet-50.json') as f:
# with open('res-sampler-color.json') as f:
    val = json.load(f)
times = val['times']
res = val['res']
# %%
dTreesNames = [
    'dTree gini1',
    'dTree gini5',
    'dTree gini10',
    'dTree gini20',
    'dTree entropy1',
    'dTree entropy5',
    'dTree entropy10',
    'dTree entropy20'
]
forestNames = [
    'Forest gini1',
    'Forest gini5',
    'Forest gini10',
    'Forest gini50',
    'Forest gini100',
    'Forest gini200',
    'Forest entropy1',
    'Forest entropy5',
    'Forest entropy10',
    'Forest entropy50',
    'Forest entropy100',
    'Forest entropy200'
]
knnNames = [ 'KNN 1', 'KNN 2', 'KNN 5', 'KNN 7', 'KNN 9' ]
svmNames = [
    'SVM linear0.1',
    'SVM linear1',
    'SVM poly0.1',
    'SVM poly1',
    'SVM rbf0.1',
    'SVM rbf1',
]
labelsNames = {
    'dTree gini1': 'G1',
    'dTree gini5': 'G5',
    'dTree gini10': 'G10',
    'dTree gini20': 'G20',
    'dTree entropy1': 'E1',
    'dTree entropy5': 'E5',
    'dTree entropy10': 'E10',
    'dTree entropy20': 'E20',
    'Forest gini1': '1G',
    'Forest gini5': '5G',
    'Forest gini10': '10G',
    'Forest gini50': '50G',
    'Forest gini100': '100G',
    'Forest gini200': '200G',
    'Forest entropy1': '1E',
    'Forest entropy5': '5E',
    'Forest entropy10': '10E',
    'Forest entropy50': '50E',
    'Forest entropy100': '100E',
    'Forest entropy200': '200E',
    'KNN 1': 'K1',
    'KNN 2': 'K2',
    'KNN 5': 'K5',
    'KNN 7': 'K7',
    'KNN 9': 'K9',
    'SVM linear0.1': 'L0.1',
    'SVM linear1': 'L1',
    'SVM poly0.1': 'P0.1',
    'SVM poly1': 'P1',
    'SVM rbf0.1': 'R0.1',
    'SVM rbf1': 'R1',
}
labelsNamesComplete = {
    'dTree gini1': 'Tree G1',
    'dTree gini5': 'Tree G5',
    'dTree gini10': 'Tree G10',
    'dTree gini20': 'Tree G20',
    'dTree entropy1': 'Tree E1',
    'dTree entropy5': 'Tree E5',
    'dTree entropy10': 'Tree E10',
    'dTree entropy20': 'Tree E20',
    'Forest gini1': 'Forest 1G',
    'Forest gini5': 'Forest 5G',
    'Forest gini10': 'Forest 10G',
    'Forest gini50': 'Forest 50G',
    'Forest gini100': 'Forest 100G',
    'Forest gini200': 'Forest 200G',
    'Forest entropy1': 'Forest 1E',
    'Forest entropy5': 'Forest 5E',
    'Forest entropy10': 'Forest 10E',
    'Forest entropy50': 'Forest 50E',
    'Forest entropy100': 'Forest 100E',
    'Forest entropy200': 'Forest 200E',
    'KNN 1': 'k-NN 1',
    'KNN 2': 'k-NN 2',
    'KNN 5': 'k-NN 5',
    'KNN 7': 'k-NN 7',
    'KNN 9': 'k-NN 9',
    'SVM linear0.1': 'SVM L0.1',
    'SVM linear1': 'SVM L1',
    'SVM poly0.1': 'SVM P0.1',
    'SVM poly1': 'SVM P1',
    'SVM rbf0.1': 'SVM R0.1',
    'SVM rbf1': 'SVM R1',
}

def getLabels(names: List[str], complete: bool = False) -> List[str]:
    return [labelsNamesComplete[n] if complete else labelsNames[n] for n in names] 


# %% plot
def subplotta(i, title, names):
    plt.subplot(4, 1, i)
    plt.tight_layout(pad=1)
    plt.title(title)
    plt.plot([res[n]['mAcc'] for n in names], '.', markersize=10)
    plt.ylabel("mAcc")
    plt.xticks(range(len(names)), getLabels(names))

def plotAccOverTime(x, names, training, fileName = None):
    y = [ res[n]['mAcc'] for n in names]
    # plt.figure(figsize=(5,4)) 
    plt.ylabel("mAcc")
    plt.xlabel("Training time [s]" if training else "Prediction time [s]")
    plt.scatter(x, y)
    for i, txt in enumerate(getLabels(names, True)):
        plt.annotate(txt, (x[i], y[i]))
    if fileName is not None:
        plt.savefig(f"charts/{fileName}.jpg")
    else:
        plt.show()
    plt.close()


# %% sttelline
plt.figure(figsize=(5,5)) 
subplotta(1, "Decision Trees", dTreesNames)
subplotta(2, "Random Forest", forestNames)
subplotta(3, "k Narest Neighbors", knnNames)
subplotta(4, "Support Vector Machine", svmNames)
# plt.show()
plt.savefig("charts/mAccShallow.jpg")
plt.close()


# %% overtime
bestPerformingTF = ['dTree gini20', 'dTree gini5', 'dTree entropy5']
bestPerformingTF += ['Forest entropy5', 'Forest entropy200', 'Forest gini5', 'Forest gini50', 'Forest gini200' ]
bestPerformingKNN = ['KNN 1', 'KNN 5','KNN 7','KNN 9']
bestPerformingTF += [ 'SVM linear0.1', 'SVM poly0.1', 'SVM rbf0.1' ]
bestPerformingKNN += [ 'SVM linear0.1', 'SVM poly0.1', 'SVM rbf0.1' ]

xDF = [ times['train'][n] for n in bestPerformingTF]
xKNN = [ times['predict'][n] for n in bestPerformingKNN]
plotAccOverTime(xDF, bestPerformingTF, True, 'overTrainingTime')
plotAccOverTime(xKNN, bestPerformingKNN, False, 'overPredictionTime')
# %%

# %%
