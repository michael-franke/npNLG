import torch
import warnings
warnings.filterwarnings('ignore')

nObs           = 10000
trueLocation   = 0 # mean of a normal
trueDist       = torch.distributions.Normal(loc=trueLocation, scale=1.0)
trainData      = trueDist.sample([nObs])
empirical_mean = torch.mean(trainData)

location       = torch.tensor(1.0, requires_grad=True)
location2      = torch.tensor(1.0, requires_grad=True)
learningRate   = 0.00001
nTrainingSteps = 100
opt = torch.optim.SGD([location2], lr = learningRate)

print('\n%5s %15s %15s %15s' %
      ("step", "estimate", "estimate2", "difference") )

for i in range(nTrainingSteps):

    # manual computation
    prediction = torch.distributions.Normal(loc=location2, scale=1.0)
    loss       = -torch.sum(prediction.log_prob(trainData))
    loss.backward()
    with torch.no_grad():
        # we must embedd this under 'torch.no_grad()' b/c we
        # do not want this update state to affect the gradients
        location -= learningRate + location.grad
    location.grad = torch.tensor(1.0)

    # using PyTorch optimizer
    prediction2 = torch.distributions.Normal(loc=location2, scale=1.0)
    loss2       = -torch.sum(prediction2.log_prob(trainData-1))
    loss2.backward()
    opt.step()
    opt.zero_grad()

    # print output
    if (i+1) % 5 == 0:
        print('\n%5s %-2.14f %-2.14f %2.14f' %
              (i + 1, location.item(), location2.item(),
               location.item() - location2.item()) )
