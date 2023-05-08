# E2E-CAL
# This file contains End-to-End training methods with constraint aware prescription layers

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# selecting the device for optimization
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# definition of neural network
class NetWind(nn.Module):
    def __init__(self, infeatures):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(infeatures, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.cplayer = build_prescriptionlayer()

    def forward(self, x):
        return self.cplayer(self.linear_relu_stack(x))


# evaluate a decision
def getcost(action, renewable_power):
    cost = action @ C
    shortcoming = 4 - action.sum() - renewable_power
    if shortcoming > 0:
        cost += shortcoming * 100
    return cost, shortcoming


# prescription layer and end-to-end loss
C = np.array([15, 20, 15, 20, 30, 25])


def build_prescriptionlayer():
    def f(x):
        scaling = torch.sigmoid(x)
        capacity = torch.Tensor(np.array([[1, 0.5, 1, 1, 1, 0.5]])).to(device)
        return scaling * capacity
    return f


def getloss(g, r, d):
    generation_cost = g @ torch.Tensor(C).to(device)
    violation = nn.functional.relu(d - g.sum(axis=1) - r)
    return generation_cost + 100 * violation


def run_myopic(testdata, loaders, length, net, resultpath, shift):
    fingerprint = time.time()
    losses = {
        'train': [],
        'test': []
    }
    optimizer = optim.Adam(net.parameters())

    for epoch in tqdm(range(100)):
        for phase in ['train', 'test']:
            runningloss = 0
            for x, y in loaders[phase]:
                if phase == 'train':
                    optimizer.zero_grad()
                output = net(x)
                loss = getloss(output, y / 1000, 4).sum()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                runningloss += loss.detach().item()
            losses[phase].append(runningloss / length[phase])
    plt.figure()
    plt.plot(losses['train'])
    plt.plot(losses['test'])
    plt.savefig(f'{resultpath}/trainingpath_{fingerprint}_shift{shift}.pdf')
    print('Test Loss: {}'.format(losses['test'][-1]))
    # evaluating the network
    with torch.no_grad():
        res = net(torch.from_numpy(testdata.drop('ActivePower', axis=1).to_numpy()).float().to(device))
    cs = [getcost(action, p / 1000) for p, action in zip(testdata.ActivePower, res.cpu().numpy())]
    costs = np.array([c[0] for c in cs])
    np.save(f'{resultpath}/{fingerprint}_shift{shift}.npy', costs)
    print('Cost with Wind Data: {}'.format(costs.mean()))
    plt.figure()
    plt.hist(costs, bins=50)
    plt.title('Production Cost')
    plt.savefig(f'{resultpath}/cost_{fingerprint}_shift{shift}.pdf')
    plt.figure()
    plt.hist([c[1] for c in cs])
    plt.title('Production Shortfall')
    plt.savefig(f'{resultpath}/shortfall_{fingerprint}_shift{shift}.pdf')


def test_myopic(df):
    for shift in [1,3,6]:
        train = df.iloc[:59533].set_index('index')
        test = df.iloc[59533:].set_index('index')
        test = test[test.index.minute%(shift*10)==0]
        traindataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(train['ActivePower'].to_numpy()).float().to(device)
        )
        testdataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(test['ActivePower'].to_numpy()).float().to(device)
        )
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=100)
        loaders = {
            'train': trainloader,
            'test': testloader
        }
        length = {
            'train': len(traindataset),
            'test': len(testdataset)
        }
        net = NetWind(3).to(device)
        run_myopic(test, loaders, length, net, 'results/e2e/cal/myopic', shift)
        
def test_myopic_stupid(df):
    for shift in [1,3,6]:
        train = df.iloc[:59533].set_index('index')[['ActivePower', 'AmbientTemperatue', 'WindDirection']]
        test = df.iloc[59533:].set_index('index')[['ActivePower', 'AmbientTemperatue', 'WindDirection']]
        test = test[test.index.minute%(shift*10)==0]
        traindataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(train['ActivePower'].to_numpy()).float().to(device)
        )
        testdataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(test['ActivePower'].to_numpy()).float().to(device)
        )
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=100)
        loaders = {
            'train': trainloader,
            'test': testloader
        }
        length = {
            'train': len(traindataset),
            'test': len(testdataset)
        }
        net = NetWind(2).to(device)
        run_myopic(test, loaders, length, net, 'results/e2e/cal/myopic_stupid', shift)


def test_historical(df):
    for shift in [1,3,6]:
        df_hist = pd.concat([df,
                            df.drop('index', axis=1).shift(1*shift).add_suffix('_1'),
                            df.drop('index', axis=1).shift(2*shift).add_suffix('_2')], axis=1)
        train = df_hist.iloc[:59533].set_index('index').dropna(axis=0)
        test = df_hist.iloc[59533:].set_index('index')
        test = test[test.index.minute%(shift*10)==0]
        traindataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(train['ActivePower'].to_numpy()).float().to(device)
        )
        testdataset = torch.utils.data.TensorDataset(
            torch.from_numpy(test.drop('ActivePower', axis=1).to_numpy()).float().to(device),
            torch.from_numpy(test['ActivePower'].to_numpy()).float().to(device)
        )
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=100)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=100)
        loaders = {
            'train': trainloader,
            'test': testloader
        }
        length = {
            'train': len(traindataset),
            'test': len(testdataset)
        }
        net = NetWind(11).to(device)
        run_myopic(test, loaders, length, net, 'results/e2e/cal/historical', shift)


if __name__ == '__main__':
    # data loading and preparation
    df = pd.read_csv('data/Turbine_Data.csv', parse_dates=True, index_col=0)
    data_available = ~(df['ActivePower'].isna() | df['AmbientTemperatue'].isna() | df['WindDirection'].isna() | df['WindSpeed'].isna())
    df = df[data_available][['ActivePower', 'AmbientTemperatue', 'WindDirection', 'WindSpeed']].reset_index()
    test_myopic(df)
    test_myopic_stupid(df)
    test_historical(df)
