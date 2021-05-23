# %%
import torch
from math import pi, sqrt, exp
import matplotlib.pyplot as plt


# %%
def generateData(n, stdX = 0, stdY = 0):
    x = (torch.arange(1, n+1) - 0.5) / n
    x += torch.randn(n) * stdX
    y = (2*pi*x).sin_()
    y += torch.randn(n)*stdY
    return (x, y)

# %%
xg, yg = generateData(100)
xt, yt = generateData(5, stdX=0.05, stdY=0.2)
plt.plot(xg, yg, '-', xt, yt, '*', linewidth=2);

# %%
def polynomial(x,w):
    X = x.view(-1,1) ** torch.arange(w.shape[0]).view(1,-1)
    return X.matmul(w)
W=torch.tensor([[0.5,0.,0.],[0.,1.,0.],[0.25,-1.,2.],[0.25,-1.,2.]]).t()
yp=polynomial(xg,W)
plt.plot(xg,yp, linewidth=2);

# %% Error
def error(yp, yt):
    return ((yp-yt)**2).mean(dim=-1)


# %% Training
def train(xt,yt, M):
    X=xt.view(-1,1)**torch.arange(M+1)
    solution, _ = torch.lstsq(yt.view(-1,1),X)
    return solution[:X.shape[1]].squeeze(-1)

trained_models = [train(xt,yt,M) for M in range(10)]
curves=[polynomial(xg,w) for w in trained_models]
curves=torch.stack(curves,dim=-1)
M_to_visualize=[0,1,3,9]
yp=curves[:,M_to_visualize]
plt.plot(xt,yt,'*',xg, yp, linewidth=2, markersize=10); 
plt.ylim([-1.5,1.5]); 

# %%
def compute_errors(x, y, trained_models):
    yp=[polynomial(x,w) for w in trained_models]
    yp=torch.stack(yp, dim=0)
    return error(yp,y)

training_errors=compute_errors(xt, yt, trained_models)
plt.plot(range(len(training_errors)), training_errors);

# %% Validation
xv,yv=generateData(10, stdX=0.05, stdY=0.2)
plt.plot(xg, yg, '-', xv, yv, '*', linewidth=2, markersize=10);


# %%
validation_errors=compute_errors(xv,yv, trained_models)
plt.plot(range(10), training_errors, '-',range(10), validation_errors,'-');
plt.ylim([-0.1,1]);


# %% Regularization
def train_reg(xt,yt, M, l):
    X=xt.view(-1,1)**torch.arange(M+1)
    X=torch.cat([X, sqrt(l)*torch.eye(M+1)],dim=0)
    yt=torch.cat([yt, torch.zeros(M+1)],dim=0)
    solution, _ = torch.lstsq(yt.view(-1,1),X)
    return solution[:X.shape[1]].squeeze(-1)
log_lambdas=[-4*j for j in range(10)] 
trained_reg_models = [train_reg(xt,yt,9,exp(l)) for l in log_lambdas]
curves_reg=[polynomial(xg,w) for w in trained_reg_models]
curves_reg=torch.stack(curves_reg,dim=-1)
idx_to_visualize=[0, 3, 6, 9]
plt.plot(xt,yt,'*',xg, curves_reg[:,idx_to_visualize], linewidth=2, markersize=10); 
plt.ylim([-1.5,1.5]); 
# %%
training_reg_errors=compute_errors(xt, yt, trained_reg_models)
plt.plot(log_lambdas, training_reg_errors);
# %%
validation_reg_errors=compute_errors(xv,yv, trained_reg_models)
plt.plot(log_lambdas, training_reg_errors, '-',log_lambdas, validation_reg_errors,'-');

# %%
from sklearn import linear_model

# %%
model_sl = linear_model.LinearRegression(fit_intercept=False)
print(model_sl)

# %% train
def train_sl(model_sl, xt,yt, M):
    X=xt.view(-1,1)**torch.arange(M+1)
    model_sl.fit(X,yt)
# %%
train_sl(model_sl, xt,yt,M=3)
print("w-parameters estimated with scikit-learn: {}".format(model_sl.coef_))
print("w-parameters estimated with PyTorch: {}".format(trained_models[3]))
# %%
