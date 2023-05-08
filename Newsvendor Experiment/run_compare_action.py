import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from newsvendor import result_analysis, or_methods, data_generation, loss_functions, training, models


class OR_True_P:
    def __init__(self, d, p, k):
        self.d = d
        self.k = k
        self.p = p

    def compute(self, P):
        '''
        Compute the optimal decision by minimizing the empirical loss
        '''
        return torch.argmax((P.cumsum(axis=1) >= (self.p-self.k)/self.p).int(), axis=1)+1

    def __call__(self, P):
        return self.compute(P)


class OR_Bayesian:
    def __init__(self, d, p, k, alpha=None):
        self.d = d
        self.k = k
        self.p = p
        self.alpha = alpha.view(1, -1) if alpha is not None else torch.ones((1, d))

    def compute(self, xis):
        '''
        Compute the optimal decision by minimizing the empirical loss
        '''
        P_hats = (torch.arange(1, self.d+1).view(1, 1, -1) == xis.unsqueeze(2)).float().mean(axis=1)
        P_post = (P_hats*xis.shape[1] + self.alpha)/(self.alpha.sum() + xis.shape[1])
        return torch.argmax((P_post.cumsum(axis=1) >= (self.p-self.k)/self.p).int(), axis=1)+1

    def __call__(self, xis):
        return self.compute(xis)


if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.mkdir('results')

    data_dirichlet = data_generation.DataDirichletPrior(11, 1000, 5, 20, alpha=torch.Tensor([0.5]*7+[2.]*4))

    # bayesian model
    model_bayesian = models.Decider(5, 0.0, 20, 20, 1, 11)
    training.train_model(model_bayesian, loss_functions.True_Loss(11, 5, 7),
                         data_dirichlet, n_epochs=1000, steplr_size=200, steplr_gamma=0.1,
                         plot_convergence=True, usetqdm=True)

    # erm model
    model_erm = models.Decider(5, 0.0, 20, 20, 1, 11)
    training.train_model(model_erm, loss_functions.ERM_Loss(11, 5, 7), data_dirichlet,
                         n_epochs=1000, steplr_size=200, steplr_gamma=0.1,
                         plot_convergence=True, usetqdm=True)

    # dro model
    model_dro = models.Decider(5, 0.0, 20, 20, 1, 11)
    training.train_model(model_dro, loss_functions.DRO_Loss(11, 5, 7, 0.25, 1e-20),
                         data_dirichlet, n_epochs=1000, steplr_size=200, steplr_gamma=0.1,
                         plot_convergence=True, usetqdm=True)

    # OR baselines
    ortrue = OR_True_P(11, 7, 5)
    orbayesian = OR_Bayesian(11, 7, 5, alpha=torch.Tensor([0.5]*7+[2.]*4))
    data_dirichlet = data_generation.DataDirichletPrior(11, 10000, 1, 20)
    xis, ps = data_dirichlet()
    oremp = or_methods.OR_Empirical(11, 7, 5)
    orkl = or_methods.OR_KLRobust(11, 7, 5, 0.25)
    analyzer = result_analysis.Analyzer(11, 7, 5)
    decision_dronet = model_dro(xis).detach().numpy().flatten()
    decision_ermnet = model_erm(xis).detach().numpy().flatten()
    decision_bayesiannet = model_bayesian(xis).detach().numpy().flatten()
    decision_dro = orkl(xis).numpy()
    decision_erm = oremp(xis).numpy()
    decision_bayesian = orbayesian(xis).numpy()
    plt.figure(figsize=(10, 10))
    nns = [decision_bayesiannet, decision_ermnet, decision_dronet]
    ors = [decision_bayesian, decision_erm, decision_dro]
    name = ['bayesian', 'erm', 'dro']
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i*3+j+1)
            ax = plt.gca()
            data = pd.DataFrame(np.vstack([nns[j], ors[i]]).T)
            data.columns = ['nn', 'or']
            sns.violinplot(data=data, x='nn', y='or', orient='h', color='C0')
            ax.invert_yaxis()
            ax.plot([1, 9], [0, 8], color='red')
            if i == 2:
                plt.xlabel('NN_'+name[j].upper())
            else:
                plt.xlabel('')
            if j == 0:
                plt.ylabel(name[i][:3].upper())
            else:
                plt.ylabel('')
            plt.xlim(0, 11)
            plt.ylim(-1, 10)
            plt.yticks(list(range(10)), list(range(1, 11)))
            plt.xticks(list(range(1, 11)), list(range(1, 11)))
    plt.savefig('results/newsvendor_decisions.pdf')
