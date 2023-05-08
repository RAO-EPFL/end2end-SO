import torch

class True_Loss:
    '''
    True Loss

    Compute the loss under the true distribution. 
    This is equivalent to minimizing the expected loss under the posterior (see paper)

    Args:
        d (int)         : the possible demands (1,2,...,d)
        cost (float)    : the cost of buying a newspaper
        revenue (float) : the revenue of selling a newspaper
    
    '''
    def __init__(self, d, cost, revenue):
        self.d = d
        self.cost = cost
        self.revenue = revenue

    def compute(self, x, sample):
        '''
        Compute the element-wise loss

        Args:
            x (torch.Tensor)    : The predicted action
            sample (tuple)      : Tuple of (xis, Ps), each torch.Tensor

        Returns:
            The sample-wise loss (torch.Tensor)
        
        '''
        xis, P = sample
        x_hat = x.view(-1,1)
        return (self.cost*x_hat-
                self.revenue*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1))*P)\
                    .sum(axis=1, keepdim=True))
    
    def __call__(self, x, sample):
        return self.compute(x, sample)

class ERM_Loss:
    '''
    ERM Loss

    Compute the loss under the sample distribution.

    Args:
        d (int)         : the possible demands (1,2,...,d)
        cost (float)    : the cost of buying a newspaper
        revenue (float) : the revenue of selling a newspaper
    
    '''
    def __init__(self, d, cost, revenue):
        self.d = d
        self.cost = cost
        self.revenue = revenue

    def compute(self, x, sample):
        '''
        Compute the element-wise loss

        Args:
            x (torch.Tensor)    : The predicted action
            sample (tuple)      : Tuple of (xis, Ps), each torch.Tensor

        Returns:
            The sample-wise loss (torch.Tensor)
        
        '''
        xis, P = sample
        P_hats = (torch.arange(1,12).view(1,1,-1)==xis.unsqueeze(2)).float().mean(axis=1)
        x_hat = x.view(-1,1)
        return (self.cost*x_hat-
                self.revenue*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1))*P_hats)\
                    .sum(axis=1, keepdim=True))

    def __call__(self, x, sample):
        return self.compute(x, sample)

class DRO_Loss:
    '''
    DRO Loss

    Compute the loss under the adversary distribution in the KL ball.

    Args:
        d (int)         : the possible demands (1,2,...,d)
        cost (float)    : the cost of buying a newspaper
        revenue (float) : the revenue of selling a newspaper
        epsilon (float) : radius of the KL ball
    
    '''
    def __init__(self, d, cost, revenue, epsilon, zero_boost):
        self.d = d
        self.cost = cost
        self.revenue = revenue
        self.epsilon = torch.Tensor([epsilon])[0]
        self.zero_boost = torch.Tensor([zero_boost])[0]

    def _get_robust_P(self, P_hats, a, verbose=False):
        '''
        Compute the KL robust distributions, which are solutions to the optimization problem
        
                    max a.T*P
                        P    s.t. P is a probability                  (1)
                                KL(P_hat, P)<=epsilon
                
        Function accepts multiple P_hat ordered in rows, with corresponding a.
        
        Args:
            P_hat (torch.Tensor):   Tensor of probabilities of shape (N, M), where N is the number 
                                    of distinct distributions and M is the number of atoms per Probability.
                                    
            a (torch.Tensor):       Tensor of coefficeints of shape (N, M), where N is the number 
                                    of distinct distributions and M is the number of distinct outcomes per Probability.
            epsilon (torch.Tensor): Dimensionless Tensor containing the radius of the uncertainty set.
        
        Returns:
            P_robust (torch.Tensor):The robust probability, solutionto problem (1)
        '''
        tol = 1e-4
        n_iter = 100
        # project the cost vectors onto the probability simplex
        n1 = torch.ones_like(a)
        pa = a-a.sum(axis=1, keepdim=True)/n1.sum(axis=1, keepdim=True)*n1
        # function handles for optimization
        k = lambda r: (P_hats/(r+pa)).sum(axis=1, keepdim=True)
        kl = lambda r: (P_hats * torch.log((r + pa) * (P_hats / (r + pa)).sum(axis=1, keepdim=True))).sum(axis=1, keepdim=True)
        
        # initial bissection interval
        R_interval = torch.hstack(((torch.exp(self.epsilon)*pa.max(axis=1, keepdim=True).values\
            -pa.min(axis=1, keepdim=True).values)\
            /(1 - torch.exp(self.epsilon)), -pa.max(axis=1, keepdim=True).values))
        f_interval = torch.hstack([kl(R_interval[:,:1]), kl(R_interval[:,1:])])
        
        # perform the optimization
        for i in range(n_iter):
            # compute the midpoint
            midpoint = R_interval.mean(axis=1, keepdim=True)
            f_midpoint = kl(midpoint)
            # create new interval
            R_interval[torch.hstack([(f_midpoint<self.epsilon), ~(f_midpoint<self.epsilon)])] = midpoint.flatten()
            f_interval[torch.hstack([(f_midpoint<self.epsilon), ~(f_midpoint<self.epsilon)])] = f_midpoint.flatten()
        if verbose and torch.any((f_interval[:,0]-f_interval[:,1]).abs()>tol):
            print('Warning: Robustness problem could not be solved to tolerance')
            print((f_interval[:,0]-f_interval[:,1]).abs().max())
        R = midpoint
        R[f_midpoint.isnan()] = R_interval[:,:1][f_midpoint.isnan()]
        return P_hats/(k(R)*(R+pa))

    def compute(self, x, sample):
        '''
        Compute the element-wise loss

        Args:
            x (torch.Tensor)    : The predicted action
            sample (tuple)      : Tuple of (xis, Ps), each torch.Tensor

        Returns:
            The sample-wise loss (torch.Tensor)
        
        '''
        xis, P = sample
        P_hats = (torch.arange(1,12).view(1,1,-1)==xis.unsqueeze(2)).float().mean(axis=1)
        x_hat = x.view(-1,1)
        A = self.cost*x_hat-self.revenue*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1)))
        P_robust = self._get_robust_P(P_hats*(1-self.zero_boost)+self.zero_boost/self.d, A)
        # if the cost is balanced, then the currentl solution (P_hat) is already optimal and my algorithm breaks
        P_robust[A.std(axis=1)==0,:] = P_hats[A.std(axis=1)==0,:]
        return (P_robust.detach()*A).sum(axis=1)

    def __call__(self, x, sample):
        return self.compute(x, sample)
    
    
class DRO_Loss_True:
    '''
    DRO Loss

    Compute the loss under the adversary distribution in the KL ball.

    Args:
        d (int)         : the possible demands (1,2,...,d)
        cost (float)    : the cost of buying a newspaper
        revenue (float) : the revenue of selling a newspaper
        epsilon (float) : radius of the KL ball
    
    '''
    def __init__(self, d, cost, revenue, epsilon, zero_boost):
        self.d = d
        self.cost = cost
        self.revenue = revenue
        self.epsilon = torch.Tensor([epsilon])[0]
        self.zero_boost = torch.Tensor([zero_boost])[0]

    def _get_robust_P(self, P_hats, a, verbose=False):
        '''
        Compute the KL robust distributions, which are solutions to the optimization problem
        
                    max a.T*P
                        P    s.t. P is a probability                  (1)
                                KL(P_hat, P)<=epsilon
                
        Function accepts multiple P_hat ordered in rows, with corresponding a.
        
        Args:
            P_hat (torch.Tensor):   Tensor of probabilities of shape (N, M), where N is the number 
                                    of distinct distributions and M is the number of atoms per Probability.
                                    
            a (torch.Tensor):       Tensor of coefficeints of shape (N, M), where N is the number 
                                    of distinct distributions and M is the number of distinct outcomes per Probability.
            epsilon (torch.Tensor): Dimensionless Tensor containing the radius of the uncertainty set.
        
        Returns:
            P_robust (torch.Tensor):The robust probability, solutionto problem (1)
        '''
        tol = 1e-4
        n_iter = 100
        # project the cost vectors onto the probability simplex
        n1 = torch.ones_like(a)
        pa = a-a.sum(axis=1, keepdim=True)/n1.sum(axis=1, keepdim=True)*n1
        # function handles for optimization
        k = lambda r: (P_hats/(r+pa)).sum(axis=1, keepdim=True)
        kl = lambda r: (P_hats * torch.log((r + pa) * (P_hats / (r + pa)).sum(axis=1, keepdim=True))).sum(axis=1, keepdim=True)
        
        # initial bissection interval
        R_interval = torch.hstack(((torch.exp(self.epsilon)*pa.max(axis=1, keepdim=True).values\
            -pa.min(axis=1, keepdim=True).values)\
            /(1 - torch.exp(self.epsilon)), -pa.max(axis=1, keepdim=True).values))
        f_interval = torch.hstack([kl(R_interval[:,:1]), kl(R_interval[:,1:])])
        
        # perform the optimization
        for i in range(n_iter):
            # compute the midpoint
            midpoint = R_interval.mean(axis=1, keepdim=True)
            f_midpoint = kl(midpoint)
            # create new interval
            R_interval[torch.hstack([(f_midpoint<self.epsilon), ~(f_midpoint<self.epsilon)])] = midpoint.flatten()
            f_interval[torch.hstack([(f_midpoint<self.epsilon), ~(f_midpoint<self.epsilon)])] = f_midpoint.flatten()
        if verbose and torch.any((f_interval[:,0]-f_interval[:,1]).abs()>tol):
            print('Warning: Robustness problem could not be solved to tolerance')
            print((f_interval[:,0]-f_interval[:,1]).abs().max())
        R = midpoint
        R[f_midpoint.isnan()] = R_interval[:,:1][f_midpoint.isnan()]
        return P_hats/(k(R)*(R+pa))

    def compute(self, x, sample):
        '''
        Compute the element-wise loss

        Args:
            x (torch.Tensor)    : The predicted action
            sample (tuple)      : Tuple of (xis, Ps), each torch.Tensor

        Returns:
            The sample-wise loss (torch.Tensor)
        
        '''
        xis, P = sample
        P_hats = (torch.arange(1,12).view(1,1,-1)==xis.unsqueeze(2)).float().mean(axis=1)
        x_hat = x.view(-1,1)
        A = self.cost*x_hat-self.revenue*(torch.minimum(x_hat, torch.arange(start=1, end=self.d+1).view(1,-1)))
        P_robust = self._get_robust_P(P*(1-self.zero_boost)+self.zero_boost/self.d, A)
        # if the cost is balanced, then the currentl solution (P_hat) is already optimal and my algorithm breaks
        P_robust[A.std(axis=1)==0,:] = P_hats[A.std(axis=1)==0,:]
        return (P_robust.detach()*A).sum(axis=1)

    def __call__(self, x, sample):
        return self.compute(x, sample)