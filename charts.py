# %% import
import json
from typing import List
import matplotlib.pyplot as plt

# %% load file
# with open('tuned-resnet-50.json') as f:
with open('res-sampler-color.json') as f:
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
numeroRighe = 4
def subplotta(i, title, names):
    plt.subplot(numeroRighe, 1, i)
    # plt.tight_layout(pad=1)
    plt.title(title)
    plt.plot([res[n]['mAcc'] for n in names], '.', markersize=10)
    plt.ylabel("mAcc")
    plt.xticks(range(len(names)), getLabels(names))

def plotAccOverTime(x, names, training, fileName = None):
    y = [ res[n]['mAcc'] for n in names]
    plt.figure(figsize=(5,5)) 
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

x = [
    [
        0.22727272727272727,
        0.3829787234042553,
        0.14473684210526316,
        0.5610687022900763,
        0.8391959798994975,
        0.2112676056338028,
        0.61003861003861,
        0.6648648648648648,
        0.21467391304347827,
        0.7890625
    ],
    [
        0.6993006993006993,
        0.7021276595744681,
        0.07894736842105263,
        0.7175572519083969,
        0.8894472361809045,
        0.676056338028169,
        0.7181467181467182,
        0.8567567567567568,
        0.3125,
        0.53125
    ],
    [
        0.7552447552447552,
        0.7659574468085106,
        0.4868421052631579,
        0.8320610687022901,
        0.7839195979899497,
        0.6830985915492958,
        0.7065637065637066,
        0.7243243243243244,
        0.5706521739130435,
        0.0234375
    ],
    [
        0.8076923076923077,
        0.851063829787234,
        0.6052631578947368,
        0.8778625954198473,
        0.7286432160804021,
        0.8838028169014085,
        0.7760617760617761,
        0.8378378378378378,
        0.35054347826086957,
        0.69140625
    ],
    [
        0.8216783216783217,
        0.8936170212765957,
        0.5263157894736842,
        0.8740458015267175,
        0.9195979899497487,
        0.9049295774647887,
        0.8648648648648649,
        0.8702702702702703,
        0.5543478260869565,
        0.76171875
    ],
    [
        0.8391608391608392,
        0.7872340425531915,
        0.35526315789473684,
        0.8702290076335878,
        0.9296482412060302,
        0.9330985915492958,
        0.8571428571428571,
        0.927027027027027,
        0.5543478260869565,
        0.765625
    ],
    [
        0.8146853146853147,
        0.7872340425531915,
        0.4868421052631579,
        0.8816793893129771,
        0.9095477386934674,
        0.9507042253521126,
        0.8494208494208494,
        0.9,
        0.6304347826086957,
        0.75390625
    ]
]
names = ["Barred Spiral", "Cigar Shaped Smooth", "Disturbed", "Edge-on with Bulge", "Edge-on without Bulge", "In-between Round Smooth", "Merging", "Round Smooth", "Unbarred Loose Spiral", "Unbarred Tight Spiral"]
l0 = []
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
l6 = []
for i in range(10):
    l0.append(x[0][i])
    l1.append(x[1][i])
    l2.append(x[2][i])
    l3.append(x[3][i])
    l4.append(x[4][i])
    l5.append(x[5][i])
    l6.append(x[6][i])


plt.plot(l0, label=names[0])
plt.plot(l1, label=names[1])
plt.plot(l2, label=names[2])
plt.plot(l3, label=names[3])
plt.plot(l4, label=names[4])
plt.plot(l5, label=names[5])
plt.plot(l6, label=names[6])

plt.legend()
plt.title("accuracy for class in resnet50")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("pippo.jpg")
# %%
