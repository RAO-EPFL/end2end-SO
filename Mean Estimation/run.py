import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# data settings
N = 20
std = 2


# Method 1: SAA
def sample_mean(samples, **kwargs):
    return samples.mean()


# Method 2: Posterior Mean
def posterior_mean(samples, prior_mean=2., prior_std=0.5, likelihood_std=std):
    posterior_mean = (prior_mean / prior_std**2 + samples.sum() / likelihood_std**2) / (1 / prior_std**2 + N / likelihood_std**2)
    posterior_std = 1/(1 / prior_std**2 + N / likelihood_std**2)
    return posterior_mean


# Method 3: NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_net(net, name):
    net = net.eval()
    with torch.no_grad():
        netpred = []
        bayesian_pred = []
        saa_pred = []
        true = []
        for i in range(5000):
            samples = torch.normal(mean=i/1000, std=torch.ones(N) * std)
            netpred.append(net(samples).numpy()[0])
            bayesian_pred.append(posterior_mean(samples).item())
            saa_pred.append(sample_mean(samples).item())
            true.append(i/1000)

    # Plotting
    plt.rcParams['text.usetex'] = True
    plt.figure()
    plt.scatter(netpred, saa_pred)
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), color='red')
    plt.xlabel(r'{\LARGE $\mu_n$}')
    plt.ylabel(r'{\LARGE $\widehat{\mu}$}')
    plt.tight_layout()
    plt.savefig(f'results/{name}_saa.pdf')

    plt.figure()
    plt.scatter(netpred, bayesian_pred)
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), color='red')
    plt.xlabel(r'{\LARGE $\mu_n$}')
    plt.ylabel(r'{\LARGE $\mu_p$}')
    plt.tight_layout()
    plt.savefig(f'results/{name}_bayesian.pdf')


def trainNet(lossf):
    net = Net()
    EPOCHS = int(5e4)
    prior_mean = 2.
    prior_std = 0.5
    likelihood_std = std
    optimizer = optim.Adam(net.parameters(),  lr=0.00005)
    losses = []

    for i in tqdm(range(EPOCHS)):
        mean = torch.normal(mean=prior_mean, std=torch.Tensor([prior_std]))
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        samples = torch.normal(mean=mean, std=likelihood_std * torch.ones((100, N)))
        pred = net(samples)
        loss = lossf(pred, torch.normal(mean=mean, std=likelihood_std*torch.ones((100, 1))), samples)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    return net


if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.mkdir('results')
    # run experiment for Algorithm 1
    net = trainNet(lambda p, m, s: ((p - m)**2).mean())
    test_net(net, 'A1')

    # run experiment for Algorithm 2
    net = trainNet(lambda p, m, s: ((p - s)**2).mean())
    test_net(net, 'A2')
