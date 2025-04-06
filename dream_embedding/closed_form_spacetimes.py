# Closed-form Spacetime Geometries for benchmarking

from layers import *
import torch.nn as nn

class DeSitterNeuralNetwork(nn.Module):
    '''
    DeSitter Neural Network Model
    '''
    def __init__(
        self,
        N, # Input Dimensionality
        D, # Space Dimensionality
        T, # Time Dimensionality
        J_encoder, # Depth of encoder
        
        ):
        super(DeSitterNeuralNetwork, self).__init__()

        self.N = N
        self.D = D
        self.T = T

        self.encoder = Encoder(N = N, hidden_dim = 100, DT = D+T, J = J_encoder)

    def compute_squared_euclidean_distance(self, X):
        K = torch.matmul(X,X.t()) 
        n = K.shape[0]
        x = torch.diagonal(K, 0).view(-1,1).repeat(1,n)
        y = torch.diagonal(K, 0).view(1,-1).repeat(n,1)
        return x + y - 2 * K        

    def pairwise_squared_euclidean_distance(self, X, Y):
        return torch.sum(X * X, dim=-1) + torch.sum(Y * Y, dim=-1) - 2 * torch.sum(X * Y, dim=-1)


    def forward(self, x, y, r=1.0):
        rsquared = r*r
        # Map node features to intermediate Euclidean space
        encoded_x = self.encoder(x)
        encoded_y = self.encoder(y)

        # Get space
        encoded_x_space = encoded_x[:,:self.D]
        encoded_y_space = encoded_y[:,:self.D]
        
        # Get time
        encoded_x_time = encoded_x[:,self.D:]
        encoded_y_time = encoded_y[:,self.D:]        
        
        encoded_x_space = r * nn.functional.normalize(encoded_x_space, dim=-1)
        encoded_y_space = r * nn.functional.normalize(encoded_y_space, dim=-1)
        
        encoded_x_space = torch.sqrt(rsquared + torch.sum(encoded_x_time * encoded_x_time, dim=-1, keepdim=True)) * encoded_x_space / r
        encoded_y_space = torch.sqrt(rsquared + torch.sum(encoded_y_time * encoded_y_time, dim=-1, keepdim=True)) * encoded_y_space / r
        
        aaa = torch.sum(encoded_x_space * encoded_y_space, dim=-1) - torch.sum(encoded_x_time * encoded_y_time, dim=-1) 
        geodesic_distance = aaa > rsquared
        aaa[geodesic_distance] = rsquared * torch.acosh(aaa[geodesic_distance] / rsquared) ** 2
        aaa[~geodesic_distance] = 2 * (aaa[~geodesic_distance] - rsquared)
        distance = aaa 

        partialorder_x = encoded_x[:,self.D:]
        partialorder_y = encoded_y[:,self.D:]

        return distance.unsqueeze(1), partialorder_x, partialorder_y

class MinkowskiNeuralNetwork(nn.Module):
    '''
    Minkowski Neural Network Model
    '''
    def __init__(
        self,
        N, # Input Dimensionality
        D, # Space Dimensionality
        T, # Time Dimensionality
        J_encoder, # Depth of encoder
        ):
        super(MinkowskiNeuralNetwork, self).__init__()

        self.N = N
        self.D = D
        self.T = T
        self.J_encoder = J_encoder
        self.encoder = Encoder(N = N, hidden_dim = 100, DT = D+T, J = J_encoder)

    def compute_squared_euclidean_distance(self, X):
        K = torch.matmul(X,X.t()) 
        n = K.shape[0]
        x = torch.diagonal(K, 0).view(-1,1).repeat(1,n)
        y = torch.diagonal(K, 0).view(1,-1).repeat(n,1)
        return x + y - 2 * K        

    def pairwise_squared_euclidean_distance(self, X, Y):
        return torch.sum(X * X, dim=-1) + torch.sum(Y * Y, dim=-1) - 2 * torch.sum(X * Y, dim=-1)


    def forward(self, x, y):

        # Map node features to intermediate Euclidean space
        encoded_x = self.encoder(x)
        encoded_y = self.encoder(y)

        # Get space
        encoded_x_space = encoded_x[:,:self.D]
        encoded_y_space = encoded_y[:,:self.D]
        

        # Get time
        encoded_x_time = encoded_x[:,self.D:]
        encoded_y_time = encoded_y[:,self.D:]        
        
        distance = self.pairwise_squared_euclidean_distance(encoded_x_space,encoded_y_space) - self.pairwise_squared_euclidean_distance(encoded_x_time,encoded_y_time)

        partialorder_x = encoded_x[:,self.D:]
        partialorder_y = encoded_y[:,self.D:]

        return distance.unsqueeze(1), partialorder_x, partialorder_y