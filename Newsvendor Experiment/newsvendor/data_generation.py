import torch

class DataDirichletPrior:
    '''
    Generate Data by Sampling Probabilities from Dirichlet Prior and sampling from the Probability

    Args:
        d (int)                         : the possible demands (1,2,...,d)
        batch_size_P (int)              : the number of different probabilities to sample
        batch_size_xi (int)             : the number of samples to generate from each probability
        N (int)                         : the number of samples contained in the feature vector
        alpha (torch.Tensor, optional)  : the concentration parameters of the distribution
    '''
    def __init__(self, d, batch_size_P, batch_size_xi, N, alpha=None):
        self.batch_size_P = batch_size_P
        self.batch_size_xi = batch_size_xi
        self.N = N
        self.prior = torch.distributions.dirichlet.Dirichlet(alpha if alpha is not None \
                                                                    else torch.ones(d))

    def sample(self):
        P_batch_values = self.prior.sample((self.batch_size_P,))
        P_batch = torch.distributions.Categorical(P_batch_values)
        xis = P_batch.sample((self.N, self.batch_size_xi))
        xis = xis.reshape(self.N,-1).T + 1
        P_batch_values = P_batch_values.repeat([self.batch_size_xi,1])
        return xis.float(), P_batch_values

    def __call__(self):
        return self.sample()

class DataBinomial:
    '''
    Generate Data by Sampling from a Binomial distribution

    Args:
        d (int)                         : the possible demands (1,2,...,d)
        psucc (float)                   : the success probability of the binomial distribution
        batch_size_P (int)              : the number of different probabilities to sample
        batch_size_xi (int)             : the number of samples to generate from each probability
        N (int)                         : the number of samples contained in the feature vector
    '''
    def __init__(self, d, psucc, batch_size, N):
        self.batch_size = batch_size
        self.N = N
        self.P = torch.distributions.Binomial(d-1, psucc)
        self.d = d

    def sample(self):
        Ps = torch.exp(self.P.log_prob(torch.arange(self.d))).repeat(self.batch_size,1)
        xis = self.P.sample((self.batch_size, self.N))+1
        return xis.float(), Ps

    def __call__(self):
        return self.sample()

class DataCustom:
    '''
    Generate Data by Sampling from a custom categorical distribution
    Warning! No sanity check is performed on `probs`.

    Args:
        probs (list-like)               : the probabilities of the Categorical distribution
        batch_size_P (int)              : the number of different probabilities to sample
        batch_size_xi (int)             : the number of samples to generate from each probability
        N (int)                         : the number of samples contained in the feature vector
    '''
    def __init__(self, probs, batch_size, N):
        self.batch_size = batch_size
        self.N = N
        self.P = torch.distributions.Categorical(torch.Tensor(probs))
        self.d = torch.Tensor(probs).shape[0]

    def sample(self):
        Ps = torch.exp(self.P.log_prob(torch.arange(self.d))).repeat(self.batch_size,1)
        xis = self.P.sample((self.batch_size, self.N))+1
        return xis.float(), Ps

    def __call__(self):
        return self.sample()
