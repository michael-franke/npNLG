import torch
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

nObs           = 10000
trueLocation   = 0
trueDist       = torch.distributions.Normal(loc=trueLocation, scale=1.0)
trainData      = trueDist.sample([nObs])

empirical_mean = torch.mean(trainData)
print('Empirical mean (mean of training data): %.5f' % empirical_mean.item())

sns.kdeplot(trainData)
plt.show()

location = torch.tensor(1.0, requires_grad=True)
print( location )

learningRate = 0.0000001
opt          = torch.optim.SGD([location], lr=learningRate)

prediction = torch.distributions.Normal(loc=location, scale=1.0)

loss     = -torch.sum(prediction.log_prob(trainData))
print(loss)

print(f"Value (initial)                = { location.item()}")
print(f"Gradient information (initial) = { location.grad}")

loss.backward()
print(f"Value (after backprop)                = { location.item()}")
print(f"Gradient information (after backprop) = { location.grad}")

opt.step()
print(f"Value (after step)                = { location.item()}")
print(f"Gradient information (after step) = { location.grad}")

opt.zero_grad()
print(f"Value (after zero-ing)                = { location.item()}")
print(f"Gradient information (after zero-ing) = { location.grad}")

nTrainingSteps= 10000
print('\n%5s %24s %15s %15s' %
      ("step", "loss", "estimate", "diff. target") )
for i in range(nTrainingSteps):
    prediction = torch.distributions.Normal(loc=location, scale=1.0)
    loss = -torch.sum(prediction.log_prob(trainData))
    loss.backward()
    if (i+1) % 500 == 0:
        print('%5d %24.3f %15.5f %15.5f' %
              (i + 1, loss.item(), location.item(), abs(location.item() - empirical_mean) ) )
        opt.step()
        opt.zero_grad()
