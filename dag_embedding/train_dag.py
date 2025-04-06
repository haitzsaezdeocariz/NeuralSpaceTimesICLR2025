# Imports
from dag_generation import *
from dag_train_functions import *
import argparse

# Add argparse setup
parser = argparse.ArgumentParser(description='Synthetic DAG Embedding')

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
    default=2500,
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

# Data Generation
parser.add_argument(
    '--num_nodes',
    type=int,
    default=50,
    help='Number of nodes in the DAG'
)

# Distance Type for DAG weights
parser.add_argument(
    '--distance_type',
    type=str,
    default='type1',
    help='What type of DAG to generate'
)

# Distance Type for DAG weights
parser.add_argument(
    '--model_type',
    type=str,
    default='nst',
    help='Specify geometry'
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

# Neural Spacetime: Time Dimension
parser.add_argument(
    '--p',
    type=int,
    default=90,
    help='Controls sparsity of the DAG'
)

args = parser.parse_args()

# Set Device
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print('Device: ',device)

# Generate Data
data_x, data_y, DAG, pos, plot_dag = generate_data_final_dag(args.num_nodes, distance_type = args.distance_type, seed = args.seed, p = args.p/100)
print(f'Data has been generated, DAG with{args.num_nodes} nodes')

# Set manual seed for reproducibility
_ = torch.manual_seed(args.seed)

# Train
average_loss, average_distortion, std_distortion, max_distortion = train(
    data_x,data_y, 
    batch_size = args.batch_size, 
    num_epochs = args.num_epochs,
    max_grad_norm = args.max_grad_norm,
    device = device,
    display_epoch = args.display_epoch,
    space_dim = args.space_dim,
    time_dim = args.time_dim,
    model_type = args.model_type,
    )
