# Neural Spacetime Model Class

from layers import *
import torch.nn as nn

class NeuralSpacetime(nn.Module):
    '''
    Neural Spacetime Model
    '''
    def __init__(
        self,
        N, # Input Dimensionality
        D, # Space Dimensionality
        T, # Time Dimensionality
        J_encoder, # Depth of encoder
        J_snowflake, # Depth of snowflake
        J_partialorder, # Depth of partial order
        
        ):
        super(NeuralSpacetime, self).__init__()

        self.N = N
        self.D = D
        self.T = T
        self.J_encoder = J_encoder
        self.J_snowflake = J_snowflake
        self.J_partialorder = J_partialorder

        self.encoder = Encoder(N = N, hidden_dim = 100, DT = D+T, J = J_encoder)
        self.snowflake = NeuralSnowflakeV2(D = D, J = J_snowflake)
        self.partialorder = PartialOrder(T = T, J = J_partialorder)
        

    def forward(self, x, y):

        # Map node features to intermediate Euclidean space
        encoded_x = self.encoder(x)
        encoded_y = self.encoder(y)

        # Get space
        encoded_x_space = encoded_x[:,:self.D]
        encoded_y_space = encoded_y[:,:self.D]

        # Compute quasi-metric distance
        distance = self.snowflake(encoded_x_space, encoded_y_space)

        # Get time
        encoded_x_time = encoded_x[:,self.D:]
        encoded_y_time = encoded_y[:,self.D:]

        # Partial order
        partialorder_x = self.partialorder(encoded_x_time)
        partialorder_y = self.partialorder(encoded_y_time)

        return distance, partialorder_x, partialorder_y