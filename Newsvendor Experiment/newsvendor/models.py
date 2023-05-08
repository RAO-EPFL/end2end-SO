import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# definition of the Decider Neural Network structure
class Decider(nn.Module):
    """
    Decider Neural Network

    Args:
        hidden_layers (int)     : Number of hidden layers
        p_dropout (float)       : Dropout Probability
        hidden_dim (int)        : Dimension of the hidden layers
        input_dim (int)         : Dimension of the input layer
        xmin (float)            : Minimum Value of the output
        xmax (float)            : Maximum Value of the output

    """
    def __init__(self, hidden_layers, p_dropout, hidden_dim, input_dim, xmin, xmax):
        self.xmin, self.xmax = xmin, xmax
        # properties of the NN
        super(Decider, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim) # 1st layer, mapping from dim 20 to hidden_dim
        self.d1 = nn.Dropout(p=p_dropout) # dropout

        self.hl = nn.ModuleList([]) # multiple hidden layers of dimension hidden_dim
        for i in range(hidden_layers):
            self.hl.append(nn.Linear(hidden_dim, hidden_dim))
        self.do = nn.ModuleList([]) # each one followed by dropout
        for i in range(hidden_layers):
            self.do.append(nn.Dropout(p=p_dropout))
        self.ol = nn.Linear(hidden_dim, 1) # linear output layer

    def forward(self, x):
        # order of operations
        y = self.d1(F.relu(self.l1(x)))
        for l, d in zip(self.hl, self.do):
            y = d(F.relu(l(y)))
        y = self.ol(y)
        return torch.sigmoid(y)*(self.xmax-self.xmin)+self.xmin