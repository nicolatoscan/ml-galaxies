# %%
import torch
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import math

# %% download
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# %% parse to torch
X=torch.from_numpy(X)
y=torch.tensor([int(d) for d in y])

# %% show stuff
num_random_digits=10
num_rows = 2
num_cols= math.ceil(num_random_digits/num_rows)
random_indices = torch.randperm(X.shape[0])[:num_random_digits]
print(f"Visualizing digits with index: {random_indices}")
for i in range(num_random_digits):
    digit_idx = random_indices[i]
    ax=plt.subplot(num_rows, num_cols, i+1)
    ax.axis('off')
    plt.imshow(X[digit_idx].reshape(28,28), cmap='gray');
    plt.title('{}'.format(y[digit_idx]));

# %%
split_sizes = [40000,15000,15000]
Xtr, Xvl, Xts = X.split(split_sizes)
ytr, yvl, yts = y.split(split_sizes)
print("Number of images: train {}, val {}, test {}".format(*[s.shape[0] for s in [Xtr, Xvl, Xts]]))
# %%
def evaluate(yt, yp, num_classes=10):
    C=(yt*num_classes+yp).bincount(minlength=num_classes**2).view(num_classes,num_classes).float()
    return {
        'Acc': C.diag().sum().item() / yt.shape[0],
        'mAcc': (C.diag()/C.sum(-1)).mean().item(),
        'mIoU': (C.diag()/(C.sum(0)+C.sum(1)-C.diag())).mean().item()
    }

# %%
import time
exec_times={}

# %% ------------------------ Decision Tree
from sklearn import tree
dtree = tree.DecisionTreeClassifier()
start=time.time()
dtree.fit(Xtr, ytr);
end=time.time()
exec_times['train']={
    'DT': end-start
}
print(f"Training time: {exec_times['train']['DT']}s")
# %% info
print(f"Tree depth: {dtree.get_depth()}")
print(f"Number of leaves: {dtree.get_n_leaves()}")
print(f"Feature used to split @ root: {dtree.tree_.feature[0]}")
print(f"Threshold used to split @ root: {dtree.tree_.threshold[0]}")
# %% plottone
feat_distrib=torch.tensor(dtree.tree_.feature)
feat_distrib=feat_distrib[feat_distrib>=0].bincount(minlength=28**2)
plt.imshow(feat_distrib.view(28,28), cmap='gray');
plt.title("Number of times a feature is used to split");
plt.show()
plt.imshow(dtree.feature_importances_.reshape(28,28), cmap='gray');
plt.title("Normalized feature importance");
plt.show()


# %% left and right
routed_left=Xtr[:,dtree.tree_.feature[0]]<=dtree.tree_.threshold[0]
left_distribution = ytr[routed_left].bincount(minlength=10)
right_distribution = ytr[~routed_left].bincount(minlength=10)
plt.bar(range(10), left_distribution-right_distribution)
plt.title('left_distribution-right_distribution')
plt.xticks(range(10));


# %% average left & right
mean_left = Xtr[routed_left].sum(0).view(28,28)
mean_right = Xtr[~routed_left].sum(0).view(28,28)

# Plots the left and right average digits
ax=plt.subplot(1,2,1)
plt.imshow(mean_left, cmap='gray')
ax.axis('off')
plt.title("Left")
ax=plt.subplot(1,2,2)
ax.axis('off')
plt.imshow(mean_right, cmap='gray')
plt.title("Right");

# %% result on training
ytr_p = dtree.predict(Xtr)

start = time.time()
yvl_p = dtree.predict(Xvl)
end = time.time()
exec_times['val']={
    'DT': end-start
}

ytr_p=torch.from_numpy(ytr_p)
yvl_p=torch.from_numpy(yvl_p)

print(f"Performance on training: {evaluate(ytr, ytr_p)}")
print(f"Performance on validation: {evaluate(yvl, yvl_p)}")
print(f"Evaluation time: {exec_times['val']['DT']}s")
# %%
def visualize_failure_cases(X, yt, yp, num_failures_to_show=6):
    failures = yt != yp
    failures_to_show = failures.nonzero()[:num_failures_to_show,0]
    for i, failure_idx in enumerate(failures_to_show):
        ax=plt.subplot(1,num_failures_to_show,i+1)
        ax.axis('off')
        plt.imshow(X[failure_idx].view(28,28),cmap='gray')
        plt.title('{} vs {}'.format(yp[failure_idx].item(),yt[failure_idx].item()))
    return failures_to_show

failures_to_show = visualize_failure_cases(Xvl, yvl, yvl_p)
# %% togliamo overfitting
dtrees = [
    tree.DecisionTreeClassifier(min_samples_leaf=ms, criterion=criterion).fit(Xtr,ytr) 
    for ms in [1,5]
    for criterion in ["gini", "entropy"]
]


# %% results
dtree_predictions_vl = [dt.predict(Xvl) for dt in dtrees]
dtree_performance_vl=[evaluate(yvl, yp) for yp in dtree_predictions_vl]
for i,metric in enumerate(['Acc', 'mAcc', 'mIoU']):
    plt.subplot(3,1,i+1)
    curve_vl=[res[metric] for res in dtree_performance_vl]
    plt.plot(curve_vl,'*',markersize=10)
    plt.grid(axis="y")
    plt.ylabel(metric)
    plt.xticks([])
# G1: criterion=gini, num_samples_leaf=1
# E1: criterion=entropy, num_samples_leaf=1
# G5: criterion=gini, num_samples_leaf=5
# E5: criterion=entropy, num_samples_leaf=5
dt_model_names=['G1', 'E1', 'G5', 'E5']
plt.xlabel('models')
plt.xticks(range(4),dt_model_names);

# %% failing on best
best_dt_model_idx = torch.tensor([res['mAcc'] for res in dtree_performance_vl]).argmax()
for i, failure_idx in enumerate(failures_to_show):
    ax=plt.subplot(1,len(failures_to_show),i+1)
    ax.axis('off')
    plt.imshow(Xvl[failure_idx].view(28,28),cmap='gray')
    plt.title('{} vs {}'.format(dtree_predictions_vl[best_dt_model_idx][failure_idx].item(),yvl[failure_idx].item()))

# %%
test_predictions_dt = dtrees[best_dt_model_idx].predict(Xts)
test_predictions_dt = torch.from_numpy(test_predictions_dt)
scores={}
scores['DT']=evaluate(yts, test_predictions_dt)
print(f"Test performance: {scores['DT']}")
# %% ------------------------ Random forest
from sklearn import ensemble
rf_model = ensemble.RandomForestClassifier(n_estimators=20, criterion="entropy")
start = time.time()
rf_model.fit(Xtr, ytr);
end = time.time()
exec_times['train']['RF']=end-start
print("Training time: {}s".format(exec_times['train']['RF']))
# %% results
rf_prediction_tr = rf_model.predict(Xtr)
start = time.time()
rf_prediction_vl = rf_model.predict(Xvl)
end = time.time()
exec_times['val']['RF']=end-start
rf_prediction_tr = torch.from_numpy(rf_prediction_tr)
rf_prediction_vl = torch.from_numpy(rf_prediction_vl)

# Print the training and validation performance
print(f"Performance on training: {evaluate(ytr, rf_prediction_tr)}")
print(f"Performance on validation: {evaluate(yvl, rf_prediction_vl)}")
print(f"Evaluation time: {exec_times['val']['RF']}s")
# %% failing
visualize_failure_cases(Xvl, yvl, rf_prediction_vl)
# %% with different number of trees
all_trees=rf_model.estimators_
rf_model.estimators_ = [all_trees[i] for i in range(2)]
curve_val={'Acc':[], 'mAcc':[], 'mIoU':[] }

for i in range(2,len(all_trees)):
    rf_model.estimators_.append(all_trees[i])
    rf_prediction_vl = rf_model.predict(Xvl)
    rf_prediction_vl = torch.from_numpy(rf_prediction_vl)
    res = evaluate(yvl, rf_prediction_vl)
    for k in res:
        curve_val[k].append(res[k])
for i,metric in enumerate(curve_val.keys()):
    plt.subplot(3,1,i+1)
    plt.plot(range(3,len(all_trees)+1), curve_val[metric], linewidth=2)
    plt.ylabel(metric)
    plt.xticks([])
    plt.grid(axis="y")
plt.xlabel('number of trees')
plt.xticks(range(3,len(all_trees)+1));

# %% On test set
rf_prediction_ts = rf_model.predict(Xts)
rf_prediction_ts = torch.from_numpy(rf_prediction_ts)
scores['RF']=evaluate(yts, rf_prediction_ts)
print("Performance on test: {}".format(scores['RF']))

# %% ------------------------ NN
from sklearn import neighbors
knn_model = neighbors.KNeighborsClassifier(n_jobs=-1)
start = time.time()
knn_model.fit(Xtr,ytr);
end = time.time()
exec_times['train']['kNN']=end-start
print("Training time: {}s".format(exec_times['train']['kNN']))
# %% random digit
query_idx = torch.randint(0,Xvl.shape[0],(1,))
prediction_knn = knn_model.predict(Xvl[query_idx])
plt.imshow(Xvl[query_idx].view(28,28), cmap='gray')
plt.title('gt {} | pred {}'.format(yvl[query_idx].item(), prediction_knn[0]))
plt.axis('off');
# %% near to random
def plot_neighbors(knn_model, query):
    neighbors_dist, neighbors_idx = knn_model.kneighbors(query)
    for i, (d, idx) in enumerate(zip(neighbors_dist[0], neighbors_idx[0])):
        ax=plt.subplot(1,len(neighbors_idx[0]), i+1)
        ax.axis('off')
        plt.imshow(Xtr[idx].view(28,28), cmap='gray')
        plt.title('{}'.format(ytr[idx]))
    print("Distances: {}".format(neighbors_dist[0]));
plot_neighbors(knn_model,Xvl[query_idx])
# %% res
knn_prediction_tr = knn_model.predict(Xtr)
start = time.time()
knn_prediction_vl = knn_model.predict(Xvl)
end = time.time()
exec_times['val']['kNN']=end-start
knn_prediction_tr = torch.from_numpy(knn_prediction_tr)
knn_prediction_vl = torch.from_numpy(knn_prediction_vl)
print(f"Performance on training: {evaluate(ytr, knn_prediction_tr)}")
print(f"Performance on validation: {evaluate(yvl, knn_prediction_vl)}")
print(f"Evaluation time: {exec_times['val']['kNN']}s")

# %% fails
failure_cases = visualize_failure_cases(Xvl, yvl, knn_prediction_vl)
# %% fails neighbors
plot_neighbors(knn_model, Xvl[failure_cases[0:1]])
# %% test
knn_prediction_ts = knn_model.predict(Xts)
knn_prediction_ts = torch.from_numpy(knn_prediction_ts)
scores['kNN']=evaluate(yts, knn_prediction_ts)
print("\nPerformance on test: {}".format(scores['kNN']))

# %% ------------------------ SVM
from sklearn import svm
svm_model = svm.SVC()
start = time.time()
svm_model.fit(Xtr, ytr);
end = time.time()
exec_times['train']['SVM']=end-start
print("Training time: {}s".format(exec_times['train']['SVM']))
# %% data
print(f"Number of support vectors per class: {svm_model.n_support_}")
# %% res
svm_prediction_tr = svm_model.predict(Xtr)
start = time.time()
svm_prediction_vl = svm_model.predict(Xvl)
end = time.time()
exec_times['val']['SVM']=end-start
svm_prediction_tr = torch.from_numpy(svm_prediction_tr)
svm_prediction_vl = torch.from_numpy(svm_prediction_vl)
print("Performance on training: {}".format(evaluate(ytr, svm_prediction_tr)))
print("Performance on validation: {}".format(evaluate(yvl, svm_prediction_vl)))
print("Evaluation time: {}s".format(exec_times['val']['SVM']))

# %% fails
failures_to_show=visualize_failure_cases(Xvl, yvl, svm_prediction_vl);

# %% other kernels
svm_models = []
for kernel in ['rbf', 'linear', 'poly']:
    for C in [0.1, 1, 10]:
        svm_models.append(svm.SVC(kernel=kernel, C=C).fit(Xtr, ytr))
        print(kernel)
        print(C)

# svm_models = [  svm.SVC(kernel=kernel, C=C).fit(Xtr, ytr) 
#                 for kernel in ['rbf', 'linear', 'poly'] 
#                 for C in [0.1, 1, 10]
#             ]
# %% other kernel results
svms_predictions_vl = [svm.predict(Xvl) for svm in svm_models]
svms_performance_vl = [evaluate(yvl, yp) for yp in svms_predictions_vl]
for i,metric in enumerate(["Acc", "mAcc", "mIoU"]):
    plt.subplot(3,1,i+1)
    curve_vl=[res[metric] for res in svms_performance_vl]
    plt.plot(curve_vl,'*',markersize=10)
    plt.ylabel(metric)
    plt.grid(axis="y")
    plt.xticks([])
svm_model_names = [
    "R.1", "R1", "R10",
    "L.1", "L1", "L10",
    "P.1", "P1", "P10",
]
plt.xticks(range(10), svm_model_names)
plt.xlabel("SVM models");