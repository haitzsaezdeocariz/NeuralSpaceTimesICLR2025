# Imports
from real_generation import *
from real_train_functions import *
from baseline_train_functions import *
import argparse

# Add argparse setup
parser = argparse.ArgumentParser(description='Real Network Embedding')

parser.add_argument(
    '--device',
    type=str,
    default='0' if torch.cuda.is_available() else 'cpu',
    help='Device to run inference on (cuda or cpu)'
)

parser.add_argument(
    '--model_type',
    type=str,
    default='min',
    help='spacetime geometry'
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
    default='Texas',
    help='Dataset'
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
if args.dataset == 'Texas':
    data_x, data_y, G = load_texas_data(seed = args.seed)
if args.dataset == 'Cornell':
    data_x, data_y, G = load_cornell_data(seed = args.seed)
if args.dataset == 'Wisconsin':
    data_x, data_y, G = load_wisconsin_data(seed = args.seed)

print(f'Data has been generated')

# Set seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Train
average_loss, average_distortion, std_distortion, max_distortion = train_baseline(
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
