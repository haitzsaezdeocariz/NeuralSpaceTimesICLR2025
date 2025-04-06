# Classes used to construct Neural Spacetimes

# Imports
import torch
import torch.nn as nn
import math

## Activation functions used by Neural Snowflake and Partial Order
class SNActivationV2(nn.Module):
    '''
    Activation function used by Neural Snowflake
    '''
    def __init__(self, last_layer = False):
        super(SNActivationV2, self).__init__()
        if last_layer:
            self.s = nn.Parameter(torch.tensor(1e-6),requires_grad = True)
            self.l = nn.Parameter(torch.tensor(1e-3),requires_grad = True)
        else:
            self.s = nn.Parameter(torch.tensor(1e-6),requires_grad = True)
            self.l = nn.Parameter(torch.tensor(1e-3),requires_grad = True)


    def forward(self, x):
        x_abs = torch.abs(x)
        sign = torch.sign(x)
        s = torch.abs(self.s)+1 # ensure exponents are always > 0
        l = torch.abs(self.l)+1
        x_1 = x_abs**s
        x_2 = x_abs**l
        output = torch.where(x_abs < 1, sign *x_1 , sign *x_2 )
        return output

class POActivation(nn.Module):
    '''
    Activation function used by Partial Order (s = l)
    '''
    def __init__(self):
        super(POActivation, self).__init__()
        self.s = nn.Parameter(torch.tensor(0.5), requires_grad = True)

    def forward(self, x):
        x_abs = torch.abs(x)
        sign = torch.sign(x)
        s = torch.abs(self.s) # ensure exponents are always > 0
        output = torch.where(x_abs < 1, sign * x_abs**s, sign * x_abs**s)
        return output
# ----------------------------------------------------------------

## Matrices
class WeightMatrix(nn.Module):
    '''
    Weight Matrix used by SN and PO, all weights are positive
    '''
    def __init__(self, size):
        super(WeightMatrix, self).__init__()
        self.size = size
        self.weight = nn.Parameter(torch.rand(size), requires_grad=True)
        self.lmbda = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.I = torch.eye(size[0], size[1], requires_grad=False, device=self.weight.device, dtype=self.weight.dtype)

    def forward(self):
        return torch.abs(self.weight) + torch.abs(self.lmbda) * self.I.to(self.weight.device)  # ensure weights are positive

class Bias(nn.Module):
    '''
    Bias terms in R (can be negative)
    '''
    def __init__(self, size):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.rand(size), requires_grad = True)

    def forward(self):
        return self.bias
# ----------------------------------------------------------------

## Networks used by Neural Spacetimes
class NeuralSnowflakeV2(nn.Module):
    '''
    Neural Snowflake V2 implementation
    Maps space dimensions to quasi-metric
    '''
    def __init__(self, D, J, hidden_dim = 10):
        super(NeuralSnowflakeV2, self).__init__()

        # Define network
        self.J = J # depth (no. layers)
        self.D = D # space dimensionality
        self.hidden_dim = hidden_dim

        # Initialize activations
        self.activation = nn.ModuleList([SNActivationV2() for _ in range(J)] + [SNActivationV2(last_layer=True)])

        # Initialize Weights
        self.weight_list = nn.ModuleList([WeightMatrix((self.D, self.hidden_dim))] + [WeightMatrix((self.hidden_dim, self.hidden_dim)) for _ in range(J-2)] + [WeightMatrix((self.hidden_dim, 1))])

        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):

        for i,w in enumerate(self.weight_list):
            d_1, d_2 = w.weight.size(0), w.weight.size(1)
            bound = 1 / (d_1*d_2)
            nn.init.uniform_(self.weight_list[i].weight, 0, bound)
            print('weights initialized')

    def forward(self, x, y):
        
        # Initial distance calculation
        u = torch.abs(self.activation[0](x) - self.activation[0](y))

        # Iterate over J
        for j in range(self.J):
            u = self.activation[j+1](u)
            W = self.weight_list[j]()
            u = torch.matmul(u,W)  # [batch,d] @ [d,d] -> [batch,d]  

        return u

class PartialOrder(nn.Module):
    '''
    Partial Order implementations
    Maps time dimensions to new time dimensions on which causality is enforced
    '''
    def __init__(self, T, J, hidden_dim = 10):
        super(PartialOrder, self).__init__()

        # Define network
        self.J = J # depth (no. layers)
        self.T = T # time dimensionality
        self.hidden_dim = hidden_dim

        # Initialize activations
        self.ReLU = torch.nn.LeakyReLU()
        self.activation = nn.ModuleList([POActivation() for _ in range(J)])

        # # Initialize Weights and Biases
        self.weight_list = nn.ModuleList([WeightMatrix((self.T, self.hidden_dim))] + [WeightMatrix((self.hidden_dim, self.hidden_dim)) for _ in range(J-2)] + [WeightMatrix((self.hidden_dim,self.T))])
        # self.weight_list = nn.ModuleList([WeightMatrix((self.T, self.T)) for _ in range(J)])
        self.bias_list = nn.ModuleList([Bias((self.hidden_dim,)) for _ in range(J-1)] + [Bias((self.T,))])

    def forward(self, x):
        u = x
        # Iterate over J
        for j in range(self.J):
            u = self.activation[j](self.ReLU(u))
            W = self.weight_list[j]()
            u = torch.matmul(u,W) + self.bias_list[j]().unsqueeze(0)  # [batch,d] @ [d,d] + [d] -> [batch,d]  

        return u

class Encoder(nn.Module):
    '''
    Encoder network
    Maps node features in R^N to R^D+T
    '''
    def __init__(self, N, hidden_dim, DT, J):
        super(Encoder, self).__init__()
        self.N = N # input dimension
        self.hidden_dim = hidden_dim # hidden dimension
        self.DT = DT # output dimension
        self.J = J # number of layers
        
        layers = []
        prev_dim = N
        for _ in range(self.J):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, DT))
        
        self.encode = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encode(x)
# ----------------------------------------------------------------