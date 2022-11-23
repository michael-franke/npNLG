import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# import sys
import warnings
warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)

##################################################
## ground-truth model
##################################################

def goalFun(x):
    return(x**3 - x**2 + 25 * np.sin(2*x))

# create linear sequence (x) and apply goalFun (y)
x = np.linspace(start = -5, stop =5, num = 1000)
y = goalFun(x)

# plot the function
d = pd.DataFrame({'x' : x, 'y' : y})
sns.lineplot(data = d, x = 'x', y = 'y')
plt.show()

##################################################
## generate training data (with noise)
##################################################

nObs = 100 # number of observations

# get noise around y observations
yNormal = torch.distributions.Normal(loc=0.0, scale=10)
yNoise  = yNormal.sample([nObs])

# get observations
xObs = 10*torch.rand([nObs])-5    # uniform from [-5,5]
yObs = xObs**3 - xObs**2 + 25 * torch.sin(2*xObs) + yNoise

# plot the data
d = pd.DataFrame({'xObs' : xObs, 'yObs' : yObs})
sns.scatterplot(data = d, x = 'xObs', y = 'yObs')
plt.show()

##################################################
## network dimension parameters
##################################################

nInput  = 1
nHidden = 10
nOutput = 1

##################################################
## trainable (matrix & slope) parameters
##           --initializing weights --
##################################################

U  = torch.tensor(np.random.rand(nHidden,nInput)   * 2 - 1,
                  requires_grad=True)
V1 = torch.tensor(np.random.rand(nHidden, nHidden) * 2 - 1,
                  requires_grad=True)
V2 = torch.tensor(np.random.rand(nHidden, nHidden) * 2 - 1,
                  requires_grad=True)
W  = torch.tensor(np.random.rand(nOutput, nHidden) * 2 - 1,
                  requires_grad=True)
b1 = torch.zeros((nHidden,1), requires_grad=True)
b2 = torch.zeros((nHidden,1), requires_grad=True)
b3 = torch.zeros((nHidden,1), requires_grad=True)

##################################################
## forward pass
##################################################

activationFun = F.relu # use ReLU fct from PyTorch

# this function takes a /single/ observation for x as input
#   and it outputs a /single/ observation for y
#   we will NOT use this one, but include for better understanding
def singleForwardPass(x):
    h1 = activationFun(U*x + b1)
    h2 = activationFun(torch.mm(V1,h1) + b2)
    h3 = activationFun(torch.mm(V2,h2) + b3)
    y  = torch.mm(W,h3)
    return(y[0,0])

# this function takes a /vector/ of observations for x as input
#   and it outputs a /vector/ of observations for y
#   we will use this function as it is massively more efficient in training
def singleForwardPassBatched(xObs):
    xObsBatched = xObs.reshape(100,1,1)       # 100 1x1 matrices
    h1 = activationFun(U @ xObsBatched + b1)  # 100 column vectors
    h2 = activationFun(V1 @ h1 + b2)
    h3 = activationFun(V2 @ h2 + b3)
    y  = W @ h3
    yReshaped = torch.reshape(y,(-1,))
    return(yReshaped)

##################################################
## optimizer & training loop
##################################################

# initialize optimizer: Adam optimizer
loss_function = nn.MSELoss()
opt = torch.optim.Adam([U,V1,V2,W,b1,b2,b3], lr=1e-4)

epochs = 100000
for i in range(epochs+1):
    if (i == 0):
        print("\n")
    yPred = singleForwardPassBatched(xObs)
    loss  = loss_function(yPred, yObs)
    # loss  = torch.mean((yPred - yObs)**2)
    if (i == 0 or i % 5000 == 0):
        print('Iteration: {:5d} | Loss: {:12}'.format(i, loss.detach().numpy().round(0)))
        # print("Iteration: ", i, " Loss: ", loss.detach().numpy())
    loss.backward()
    opt.step()
    opt.zero_grad()

yPred = singleForwardPassBatched(xObs)

# plot the data
d = pd.DataFrame({'xObs' : xObs.detach().numpy(),
                  'yObs' : yObs.detach().numpy(),
                  'yPred': yPred.detach().numpy()})
dWide = pd.melt(d, id_vars = 'xObs', value_vars= ['yObs', 'yPred'])
sns.scatterplot(data = dWide, x = 'xObs', y = 'value', hue = 'variable', alpha = 0.7)
x = np.linspace(start = -5, stop =5, num = 1000)
y = goalFun(x)
plt.plot(x,y, color='g', alpha = 0.5)
plt.show()
