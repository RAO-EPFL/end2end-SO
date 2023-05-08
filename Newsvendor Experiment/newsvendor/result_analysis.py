import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Analyzer:
    def __init__(self, d, p, k):
        self.d = d
        self.p = p
        self.k = k

    def _decision_loss(self, x, P):
        x_hat = x.view(-1,1)
        return (self.k*x_hat
                -self.p*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1))
                *P).sum(axis=1, keepdim=True))

    def plot_decisions(self, Ps, x_hat):
        losses = self._decision_loss(x_hat, Ps)
        plt.subplot(2,2,1)
        lossf = self._decision_loss(torch.arange(1,12).view(-1,1), Ps[0,:].view(1,-1)).flatten()
        plt.plot(list(range(1,12)),lossf)
        plt.vlines(x_hat.flatten().detach().numpy(), min(lossf), max(lossf), alpha=0.2, color='C1')
        plt.subplot(2,2,2)
        plt.hist(losses.flatten().detach().numpy(), label=f'Mean Loss: {losses.mean()}')
        plt.legend()
    
    def plot_cdf(self, ps, *x_hats, log=True):
        #plt.figure(figsize=(8,6))
        for i, x_hat in enumerate(x_hats):
            losses = self._decision_loss(x_hat[1], ps).detach().numpy().flatten()
            df = pd.DataFrame(data={
                'loss' : -losses,
                'number' : 1
            }).sort_values('loss')
            df['sums'] = df.number.cumsum()
            print(losses)
            if x_hat[0]=='True':
                l =round(float(-losses.mean()),2)
                plt.fill_between(df.loss, df.sums/df.shape[0], label=f'{x_hat[0]}: {l}', alpha=0.2, color='grey')
                plt.vlines(-losses.mean(), 1e-4, 1, colors=['grey'])
            else:
                l = round(float(-losses.mean()),2)
                plt.plot(df.loss, df.sums/df.shape[0], label=f'{x_hat[0]}: {l}', color=f'C{i}')
                plt.vlines(-losses.mean(), 1e-4, 1, colors=[f'C{i}'], linestyles='dashed')
        plt.legend()
        plt.title('Profit Distribution')
        plt.xlabel('Profit')
        if log:
            plt.yscale('log')