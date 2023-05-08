import torch
import numpy as np

class OR_Empirical:
    def __init__(self, d, p, k):
        self.d = d
        self.k = k
        self.p = p
        
    def compute(self, xis):
        '''
        Compute the optimal decision by minimizing the empirical loss
        '''
        P_hats = (torch.arange(1, self.d+1).view(1,1,-1)==xis.unsqueeze(2)).float().mean(axis=1)
        return torch.argmax((P_hats.cumsum(axis=1)>=(self.p-self.k)/self.p).int(), axis=1)+1
    
    def __call__(self, xis):
        return self.compute(xis)

class OR_KLRobust:
    def __init__(self, d, p, k, r):
        self.d = d
        self.k = k
        self.p = p
        self.r = r

    def DRO_c_hat(self, P_hat):
        def decision_loss(x, P):
            x_hat = x.view(-1,1)
            return (self.k*x_hat
                    -self.p*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1))
                    *P).sum(axis=1, keepdim=True))

        def gamma(x, xi):
            return self.k*x-self.p*torch.minimum(x, xi)

        xgrid = torch.arange(1, self.d+1)
        c_robust = []
        for x in xgrid:
            alpha_min = gamma(x, torch.Tensor([0])[0])
            alpha_max = ((alpha_min - np.exp(-self.r) * decision_loss(x, P_hat)) / (1 - np.exp(-self.r)))[0,0]
            alpha_grid = torch.linspace(alpha_min, alpha_max, 10)
            fval = alpha_grid-np.exp(-self.r)*((alpha_grid.view(-1,1) - gamma(x, torch.arange(1, self.d+1).view(1,-1))) ** P_hat).prod(axis=1)
            c_robust.append(fval.min().numpy())
        return c_robust
        
    def compute(self, xis):
        '''
        Compute the optimal decision by minimizing the robust loss
        '''
        P_hat = (xis.view(-1,1) == torch.arange(1,self.d+1).view(1,-1)).float().mean(axis=0)
        return np.argmin(self.DRO_c_hat(P_hat))+1

    def __call__(self, xis):
        return torch.Tensor([self.compute(xi) for xi in xis])