# Imports
from dream_graph_generation import *
from dream_train_functions import *
import argparse

# Add argparse setup
parser = argparse.ArgumentParser(description='Dream5 Network Embedding')

parser.add_argument(
    '--device',
    type=str,
    default='0' if torch.cuda.is_available() else 'cpu',
    help='Device to run inference on (cuda or cpu)'
)

# Training setup
parser.add_argument(
    '--batch_size',
    type=int,
    default=10000,
    help='Training Batch Size'
)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=30000,
    help='Training No. Epochs'
)
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='Maximum Gradient for Clipping'
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
    help='Learning rate'
)
parser.add_argument(
    '--display_epoch',
    type=int,
    default=100,
    help='Display Results Every X Epochs'
)
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    help='Fix seed for reproducibility'
)

parser.add_argument(
    '--dataset',
    type=str,
    default='silico',
    help='Dataset'
)

parser.add_argument(
    '--model_type',
    type=str,
    default='nst',
    help='Geometry'
)

# Neural Spacetime: Space Dimension
parser.add_argument(
    '--space_dim',
    type=int,
    default=10,
    help='Space dimensionality for neural snowflake v2'
)

# Neural Spacetime: Time Dimension
parser.add_argument(
    '--time_dim',
    type=int,
    default=10,
    help='Time dimensionality for neural partial order'
)

args = parser.parse_args()

# Set Device
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print('Device: ',device)

# Generate Data
data_x, data_y, G = load_dream_data(dataset = args.dataset, seed = args.seed)

print(f'Data has been generated')

# Set manual seed for reproducibility
_ = torch.manual_seed(args.seed)

# Train
average_loss, average_distortion, std_distortion, max_distortion = train(
    data_x,data_y, 
    dataset = args.dataset,
    batch_size = args.batch_size, 
    num_epochs = args.num_epochs,
    max_grad_norm = args.max_grad_norm,
    device = device,
    display_epoch = args.display_epoch,
    space_dim = args.space_dim,
    time_dim = args.time_dim,
    model_type = args.model_type,
    )
