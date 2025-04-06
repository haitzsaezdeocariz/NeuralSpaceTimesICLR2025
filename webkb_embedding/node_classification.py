import torch
import torch.nn.functional as F
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import MLP, GCNConv, GATConv
from torch_geometric.nn.conv import ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import argparse
import warnings
warnings.filterwarnings("ignore")


# Import Neural Spacetime model
from neural_spacetime import *

# Set random seed for reproducibility
torch.manual_seed(123)


# Argument parser
parser = argparse.ArgumentParser(description="Node Classification with GCN or MLP")
parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT', 'MLP', 'CPGNNMLP', 'CPGNNCheby'],
                    help="Type of model to use: 'GCN', 'MLP', 'CPGNNMLP', 'GAT', 'CPGNNCheby' ")
parser.add_argument('--preprocess', action='store_true',
                    help="Enable preprocessing with Neural Spacetime")
parser.add_argument('--hidden_dim', type=int, default=20,
                    help="Dimension of hidden layers")
parser.add_argument('--datasetname', type=str, default='Cornell',
                    help="Dataset name: 'Cornell', 'Texas', 'Wisconsin'")
args = parser.parse_args()

# Load dataset
dataset = WebKB(root=f'/tmp/{args.datasetname}', name=f'{args.datasetname}', transform=NormalizeFeatures())
data = dataset[0]

# Define model classes
class NodeClassificationMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = MLP(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=2
        )
    
    def forward(self, x):
        return self.mlp(x)

class NodeClassificationGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class NodeClassificationGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class CPGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, estimator_type='MLP', 
                 lambda_p=5e-4, eta=0.01, beta1_iterations=100):
        super(CPGNN, self).__init__()
       
        # Hyperparameters
        self.lambda_p = lambda_p  # L2 regularization weight
        self.eta = eta  # Regularization weight for compatibility matrix
        self.beta1_iterations = beta1_iterations  # Pretraining iterations
        self.num_classes = num_classes
       
        # Prior Belief Estimator (Stage S1)
        if estimator_type == 'MLP':
            self.prior_estimator = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif estimator_type == 'Cheby':
            self.cheb1 = ChebConv(input_dim, hidden_dim, K=2)
            self.relu = nn.ReLU()
            self.cheb2 = ChebConv(hidden_dim, num_classes, K=2)

        self.estimator_type = estimator_type
       
        # Compatibility Matrix for Propagation (Stage S2)
        self.compatibility_matrix = nn.Parameter(torch.randn(num_classes, num_classes))
       
        self.num_layers = num_layers
    
    def pretrain_prior_estimator(self, x, y_train, train_mask, edge_index=None):
        """
        Pretrain the prior belief estimator.
        """
        if self.estimator_type == 'MLP':
            optimizer = torch.optim.AdamW(self.prior_estimator.parameters(), weight_decay=self.lambda_p, lr=lr)
        elif self.estimator_type == 'Cheby':
            optimizer = torch.optim.AdamW(
                list(self.cheb1.parameters()) + 
                list(self.relu.parameters()) + 
                list(self.cheb2.parameters()), 
                weight_decay=self.lambda_p,
                lr=lr,
            )
        
        for _ in range(self.beta1_iterations):
            optimizer.zero_grad()
            prior_beliefs = self.get_prior_beliefs(x, edge_index)
            loss = F.cross_entropy(prior_beliefs[train_mask], y_train[train_mask])
            loss.backward()
            optimizer.step()
        
        return prior_beliefs
    
    def get_prior_beliefs(self, x, edge_index=None):
        """
        Compute prior beliefs using the chosen estimator type.
        """
        if self.estimator_type == 'Cheby':
            prior_beliefs = self.cheb1(x, edge_index) 
            prior_beliefs = self.relu(prior_beliefs)
            prior_beliefs = self.cheb2(prior_beliefs, edge_index) 
            
        elif self.estimator_type == 'MLP':
            prior_beliefs = self.prior_estimator(x)
        
        return prior_beliefs
    
    def initialize_compatibility_matrix(self, x, y_train, train_mask, edge_index):
        y_train_one_hot = F.one_hot(y_train, num_classes=self.num_classes).to(dtype=x.dtype, device=x.device)
        """
        Initialize compatibility matrix using Sinkhorn-Knopp algorithm.
        """
        # Compute prior beliefs 
        prior_beliefs = F.softmax(self.get_prior_beliefs(x, edge_index), dim=-1)
        
        # Create mask matrix M: 1 for training nodes, 0 for others
        # Create matrix M
        M = torch.zeros((x.shape[0], self.num_classes), dtype=prior_beliefs.dtype, device=prior_beliefs.device)
        M[train_mask] = 1.0
        
        # Compute enhanced belief matrix B˜ 
        # B˜ = Ytrain + (1 − M) ◦ Bp
        enhanced_beliefs = y_train_one_hot + (1 - M) * prior_beliefs
        
        # Compute adjacency matrix (simple version)
        adj_matrix = to_dense_adj(edge_index)[0]
        
        # Compute input to Sinkhorn-Knopp: Y^T * A * B˜ 
        input_to_SK = torch.mm(y_train_one_hot.T, torch.mm(adj_matrix, enhanced_beliefs))
        
        # Compute initial compatibility matrix estimate
        H_hat = self.sinkhorn_knopp(input_to_SK)
        
        # Initialize compatibility matrix 
        init_matrix = 0.5 * (H_hat + H_hat.T) - 1 / self.num_classes
        
        # Set as initial compatibility matrix
        with torch.no_grad():
            self.compatibility_matrix.copy_(init_matrix)
    
    def sinkhorn_knopp(self, A, num_iterations=20):
        """
        Sinkhorn-Knopp algorithm to make a matrix doubly stochastic.
        Ensures rows and columns sum to 1.
        
        Args:
            A (torch.Tensor): Input matrix (non-negative).
            num_iterations (int): Number of iterations to perform.
        
        Returns:
            torch.Tensor: Doubly stochastic matrix.
        """
        A = torch.abs(A)  # Ensure non-negativity
        for _ in range(num_iterations):
            row_sums = A.sum(dim=1, keepdim=True)
            A = A / (row_sums + 1e-8)  # Avoid division by zero for zero rows
            col_sums = A.sum(dim=0, keepdim=True)
            A = A / (col_sums + 1e-8)  # Avoid division by zero for zero columns
        return A
    
    def compatibility_matrix_regularization(self):
        """
        Regularization term to keep the compatibility matrix centered.
        Computes Φ(H¯) = η * Σ_i |Σ_j H¯_ij|.
        """
        row_sums = torch.sum(self.compatibility_matrix, dim=1)  # Sum over columns (j) for each row (i)
        return self.eta * torch.sum(torch.abs(row_sums))
    
    def forward(self, x, edge_index):
        adj_matrix = to_dense_adj(edge_index)[0]

        # Stage S1: Prior Belief Estimation
        prior_beliefs = F.softmax(self.get_prior_beliefs(x, edge_index), dim=-1)
        
        # Center the prior beliefs
        centered_beliefs = prior_beliefs - (1 / self.num_classes)
        
        # Stage S2: Compatibility-guided Propagation
        current_beliefs = centered_beliefs.clone()
        for _ in range(self.num_layers):
            # Propagate beliefs using compatibility matrix
            current_beliefs = centered_beliefs + torch.mm(adj_matrix, torch.mm(current_beliefs, self.compatibility_matrix))
        
        # Final beliefs
        final_beliefs = current_beliefs
        
        return final_beliefs
    
    def loss_function(self, predictions, labels, prior_estimator_params):
        """
        Compute the complete loss function for CPGNN.
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(predictions, labels)
        
        # Compatibility matrix regularization
        matrix_reg = self.compatibility_matrix_regularization()
        
        # Total loss 
        total_loss = ce_loss + matrix_reg
        
        return total_loss

# Function to get masks for a specific fold
def get_masks(data, fold=0):
    train_mask = data.train_mask[:, fold].bool()
    val_mask = data.val_mask[:, fold].bool()
    test_mask = data.test_mask[:, fold].bool()
    return train_mask, val_mask, test_mask

# Training and testing functions
def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Different forward pass for MLP and GCN
    if isinstance(model, NodeClassificationMLP):
        out = model(data.x)
    else:
        out = model(data.x, data.edge_index)
    
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Different forward pass for MLP and GCN
        if isinstance(model, NodeClassificationMLP):
            out = model(data.x)
        else:
            out = model(data.x, data.edge_index)
        
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc

# Hyperparameters
lr = 0.01
epochs = 400

if args.preprocess:
    with torch.no_grad():
        model_features = 20 + data.num_features
        data_features = data.x.cuda() 
        NST = torch.load(f'NST_{args.datasetname}_10_10.pt')
        NST = NST.cuda()
        x_hat = NST.encoder(data_features)
        t = NST.partialorder(x_hat[:,NST.D:])
        data.x = torch.cat((x_hat[:,:NST.D],t,data_features),dim=-1).cpu()
else:
    model_features = data.num_features

# Run cross-validation
results = []
for fold in range(10):
    # Get masks for current fold
    train_mask, val_mask, test_mask = get_masks(data, fold)
    
    # Initialize model based on type
    if args.model_type == 'MLP':
        model = NodeClassificationMLP(
            in_channels=model_features, 
            hidden_channels=args.hidden_dim, 
            out_channels=dataset.num_classes
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Standard training loop
        for epoch in range(epochs):
            loss = train(model, data, train_mask, optimizer, criterion)
    
    elif args.model_type == 'GCN':
        model = NodeClassificationGCN(
            in_channels=model_features, 
            hidden_channels=args.hidden_dim, 
            out_channels=dataset.num_classes
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Standard training loop
        for epoch in range(epochs):
            loss = train(model, data, train_mask, optimizer, criterion)

    elif args.model_type == 'GAT':
        model = NodeClassificationGAT(
            in_channels=model_features, 
            hidden_channels=args.hidden_dim, 
            out_channels=dataset.num_classes
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Standard training loop
        for epoch in range(epochs):
            loss = train(model, data, train_mask, optimizer, criterion)
    
    elif args.model_type in ['CPGNNMLP', 'CPGNNCheby']:
        # Determine estimator type
        estimator_type = 'MLP' if args.model_type == 'CPGNNMLP' else 'Cheby'
        
        # Initialize CPGNN model
        model = CPGNN(
            input_dim=model_features, 
            hidden_dim=args.hidden_dim, 
            num_classes=dataset.num_classes,
            estimator_type=estimator_type,
            lambda_p=0.001,  # L2 regularization weight
            eta=0.01,         # Compatibility matrix regularization
            beta1_iterations=50  # Pretraining iterations
        )
        
        # Pretrain prior estimator
        model.pretrain_prior_estimator(data.x, data.y, train_mask, data.edge_index)
        print('Prior believes have been pretrained')
        
        # Initialize compatibility matrix
        model.initialize_compatibility_matrix(data.x, data.y, train_mask, data.edge_index)
        
        # Optimizer and training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Custom loss function
            loss = model.loss_function(
                out[train_mask], 
                data.y[train_mask], 
                list(model.prior_estimator.parameters()) if estimator_type == 'MLP' 
                else list(model.cheb1.parameters()) + list(model.cheb2.parameters())
            )
            
            loss.backward()
            optimizer.step()
    
    # Evaluate this fold
    val_acc = test(model, data, val_mask)
    test_acc = test(model, data, test_mask)
    results.append((val_acc, test_acc))
    # print(f'Fold {fold}: Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}')

# Print cross-validation summary
print(f"\nCross-Validation Summary for {args.model_type}:")
val_accs, test_accs = zip(*results)
print(f"Mean Val Accuracy: {torch.tensor(val_accs).mean():.4f} ± {torch.tensor(val_accs).std():.4f}")
print(f"Mean Test Accuracy: {torch.tensor(test_accs).mean():.4f} ± {torch.tensor(test_accs).std():.4f}")