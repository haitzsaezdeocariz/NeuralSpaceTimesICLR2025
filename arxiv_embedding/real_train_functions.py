# Generic Imports 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import time
from torch.utils.data import TensorDataset, DataLoader

# Import Neural Spacetime model
from neural_spacetime import *

# training function
def train(
        data_x,
        data_y, 
        dataset = 'Arxiv',
        model_type ='nst', 
        batch_size = 2500, 
        num_epochs = 30000, 
        max_grad_norm = 1.0, 
        device = 'cpu', 
        display_epoch = 10,
        space_dim = 10,
        time_dim = 10,
        ):

    if dataset == 'Arxiv':
        feature_dim = 128

    # Load models
    if model_type == 'nst':
        metric = NeuralSpacetime(
            N = feature_dim, #input dimension
            D = space_dim, # Space Dimensionality
            T = time_dim, # Time Dimensionality
            J_encoder = 10, # Depth of encoder
            J_snowflake = 4, # Depth of snowflake
            J_partialorder = 4, # Depth of partial order
        )

        # Move to GPU
        metric = metric.to(device)

        # Display network for reference
        print(metric)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(count_parameters(metric))
        
    # Loss function for space embedding
    criterion_space = nn.MSELoss()

    def criterion_time(x_u,x_v):

        # Compute loss (ReLU is unstable) 
        sigmoid = torch.nn.Sigmoid()
        diff = x_u-x_v
        sigmoid_diff = sigmoid(10*diff) # steep sigmoid
        sigmoid_diff_sum = torch.mean(sigmoid_diff,dim=-1)
        loss = torch.mean(sigmoid_diff_sum)


        # Compute correctness
        relu = torch.nn.ReLU()
        relu_diff = relu(diff)
        relu_diff_sum = torch.sum(relu_diff,dim=-1)
        zero_mask = relu_diff_sum == 0
        num_zeros = torch.sum(zero_mask).item()
        total_correct = num_zeros/relu_diff_sum.shape[0]

        loss = loss * (1-total_correct)

        return loss, total_correct

    # Optimizer
    optimizer = optim.AdamW(metric.parameters(), lr=1e-4)

    # Set up dataloader
    train_dataset = TensorDataset(data_x, data_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        
        # Running loss and distortion
        running_loss = 0.0
        running_loss_space = 0.0
        running_loss_time = 0.0
        distortion = []
        
        # Iterate for each epoch over the train dataloader
        for i, data in enumerate(train_loader):

            # Get inputs and labels from the data loader
            inputs, labels = data
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Compute distance and partial order embedding
            distance, partialorder_x, partialorder_y = metric(inputs[:, feature_dim:],inputs[:, :feature_dim])
            distance = distance.squeeze(1)

            # Create a mask based on connectivity
            connected = (labels != 0)

            # Apply the mask to both distance and label distances
            masked_distance = distance[connected]
            masked_labels = labels[connected]

            # Calculate the spatial loss only on non-zero entries
            loss_space = criterion_space(masked_distance, masked_labels)

            # Calculate the time loss 
            masked_partialorder_x = partialorder_x[connected]
            masked_partialorder_y = partialorder_y[connected]
            loss_time, total_correct = criterion_time(masked_partialorder_x,masked_partialorder_y)

            # Total loss
            loss = loss_time + loss_space

            # Store distortion
            distortion_batch = masked_labels/masked_distance
            distortion_batch = distortion_batch.unsqueeze(1)
            distortion.append(masked_labels/masked_distance)

            # Backward pass and optimize
            loss.backward()

            # # Clip gradients to prevent exploding gradients
            clip_grad_norm_(metric.parameters(), max_grad_norm)

            # Optimizer Step
            optimizer.step()
            # print(metric.encoder.encode[0].weight[0])

            # Update the running loss
            running_loss += loss.item()
            running_loss_space += loss_space.item()
            running_loss_time += loss_time.item()

        # Compute epoch statistics
        average_loss = running_loss/len(train_loader)
        average_loss_space = running_loss_space/len(train_loader)
        average_loss_time = running_loss_time/len(train_loader)
        distortion = torch.cat(distortion)
        average_distortion = torch.mean(distortion).item()
        std_distortion = torch.std(distortion).item()
        max_distortion = torch.max(distortion).item()
        
        if (epoch + 1) % display_epoch == 0:
            print(f'Epoch {epoch+1} loss: ',average_loss)
            print(f'Epoch {epoch+1} space loss: ',average_loss_space)
            print(f'Epoch {epoch+1} time loss: ',average_loss_time)
            print(f'Epoch {epoch+1} avg distortion: ',average_distortion)
            print(f'Epoch {epoch+1} std distortion: ',std_distortion)
            print(f'Epoch {epoch+1} max distortion: ',max_distortion)
            print(f'Epoch {epoch+1} directionality encoded: ',total_correct)
            print('--------------------------')

    # Return statistic for last epoch
    return average_loss, average_distortion, std_distortion, max_distortion