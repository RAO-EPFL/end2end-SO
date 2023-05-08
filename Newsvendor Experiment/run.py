import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from torch import distributions
from newsvendor import result_analysis, or_methods, data_generation, loss_functions, training, models

if not os.path.isdir('results'):
    os.mkdir('results')

# training data
data_dirichlet = data_generation.DataDirichletPrior(11, 1000, 5, 20)

# Bayesian Model
model_bayesian = models.Decider(5, 0.0, 20, 20, 1, 11)
training.train_model(model_bayesian, loss_functions.True_Loss(11, 5, 7), data_dirichlet, n_epochs=1000, steplr_size=1000, steplr_gamma=0.75, plot_convergence=False, usetqdm=True);

# DRO
model_dro = models.Decider(5, 0.0, 20, 20, 1, 11)
training.train_model(model_dro, loss_functions.DRO_Loss(11, 5, 7, 0.025, 1e-3), data_dirichlet, n_epochs=1000, steplr_size=1000, steplr_gamma=0.75, plot_convergence=False, usetqdm=True);

# ERM
model_erm = models.Decider(5, 0.0, 20, 20, 1, 11)
training.train_model(model_erm, loss_functions.ERM_Loss(11, 5, 7), data_dirichlet, n_epochs=1000, steplr_size=1000, steplr_gamma=0.75, plot_convergence=False, usetqdm=True);

# Result Analysis
import torch
class OR_True_P:
    def __init__(self, d, p, k):
        self.d = d
        self.k = k
        self.p = p

    def compute(self, P):
        '''
        Compute the optimal decision by minimizing the empirical loss
        '''
        return torch.argmax((P.cumsum(axis=1)>=(self.p-self.k)/self.p).int(), axis=1)+1

    def __call__(self, P):
        return self.compute(P)
    
class OR_Bayesian:
    def __init__(self, d, p, k, alpha=None):
        self.d = d
        self.k = k
        self.p = p
        self.alpha = alpha.view(1,-1) if alpha is not None else torch.ones((1,d))
        
    def compute(self, xis):
        '''
        Compute the optimal decision by minimizing the empirical loss
        '''
        P_hats = (torch.arange(1, self.d+1).view(1,1,-1)==xis.unsqueeze(2)).float().mean(axis=1)
        P_post = (P_hats*xis.shape[1] + self.alpha)/(self.alpha.sum() + xis.shape[1])
        return torch.argmax((P_post.cumsum(axis=1)>=(self.p-self.k)/self.p).int(), axis=1)+1
    
    def __call__(self, xis):
        return self.compute(xis)

ortrue = OR_True_P(11,7,5)
orbayesian = OR_Bayesian(11,7,5)

# Comparison
data_dirichlet = data_generation.DataDirichletPrior(11, 10000, 1, 20)
xis, ps = data_dirichlet()
oremp = or_methods.OR_Empirical(11,7,5)
orkl = or_methods.OR_KLRobust(11,7,5, 0.025)
analyzer = result_analysis.Analyzer(11, 7, 5)

plt.figure()
analyzer.plot_cdf(ps, ('ERM', oremp(xis)), ('Bayesian', orbayesian(xis)), ('DRO', orkl(xis)), ('NN_ERM', model_erm(xis)), ('NN_BAY', model_bayesian(xis)), ('NN_DRO', model_dro(xis)), ('True', ortrue(ps)))
plt.savefig('results/cdf_comparison_same_P.pdf')

data_dirichlet1 = data_generation.DataDirichletPrior(11, 10000, 1, 20, alpha=torch.Tensor([0.1]*1+[0.1]*4+[2.]*6))
xis, ps = data_dirichlet1()
plt.figure()
analyzer.plot_cdf(ps, ('ERM', oremp(xis)), ('Bayesian', orbayesian(xis)), ('DRO', orkl(xis)), ('NN_ERM', model_erm(xis)), ('NN_BAY', model_bayesian(xis)), ('NN_DRO', model_dro(xis)), ('True', ortrue(ps)))
plt.savefig('results/cdf_comparison_different1_P.pdf')
distributions.kl.kl_divergence(data_dirichlet1.prior, data_dirichlet.prior)

data_dirichlet2 = data_generation.DataDirichletPrior(11, 10000, 1, 20, alpha=torch.Tensor([2.]*6+[0.1]*5))
xis, ps = data_dirichlet2()
plt.figure()
analyzer.plot_cdf(ps, ('ERM', oremp(xis)), ('Bayesian', orbayesian(xis)), ('DRO', orkl(xis)), ('NN_ERM', model_erm(xis)), ('NN_BAY', model_bayesian(xis)), ('NN_DRO', model_dro(xis)), ('True', ortrue(ps)))
plt.savefig('results/cdf_comparison_different2_P.pdf')
distributions.kl.kl_divergence(data_dirichlet2.prior, data_dirichlet.prior)