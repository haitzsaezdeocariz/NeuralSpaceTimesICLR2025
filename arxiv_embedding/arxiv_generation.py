# Import
import networkx as nx
import numpy as np
import torch
import random
import os
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.utils import to_networkx
import pickle

def load_ogbn_arxiv_data(seed=None, data_dir='data'):
    """
    Load and process the ogbn-arxiv dataset. If preprocessed data exists, load it directly.
    Otherwise, process and save it.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define file paths
    data_x_path = os.path.join(data_dir, f'arxiv_data_x_seed_{seed}.pt')
    data_y_path = os.path.join(data_dir, f'arxiv_data_y_seed_{seed}.pt')
    graph_path = os.path.join(data_dir, f'arxiv_graph_seed_{seed}.pkl')
    
    # Check if preprocessed data exists
    if os.path.exists(data_x_path) and os.path.exists(data_y_path) and os.path.exists(graph_path):
        print("Loading preprocessed arxiv data...")
        data_x = torch.load(data_x_path)
        data_y = torch.load(data_y_path)

        # Load the graph using pickle
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        print("Arxiv data loaded successfully!")
        return data_x, data_y, G
    
    print("Processing arxiv data for the first time...")
    
    def calculate_edge_weights(G, node_features):
        """
        Calculate cosine similarities only for existing edges
        """
        for u, v in G.edges():
            u_features = node_features[u]
            v_features = node_features[v]

            cos_sim = torch.nn.functional.cosine_similarity(
                u_features, 
                v_features, 
                dim=0
            ).item()
            cos_sim = max(cos_sim, 0.01)
            G[u][v]['weight'] = cos_sim

    # Set seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # Load the ogbn-arxiv dataset
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogbn-arxiv')
    graph, _ = dataset[0]  # We don't need labels for this task
    
    # Get node features and edges
    node_features = torch.tensor(graph['node_feat'], dtype=torch.float32)
    edge_index = graph['edge_index'].T
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(len(node_features)):
        G.add_node(i)
    
    # Add edges and remove edges with high cosine similarity
    similarity_threshold = 0.99  # You can adjust this threshold as needed
    rejected_edges = 0  # Counter for rejected edges

    for edge in edge_index:
        u, v = int(edge[0]), int(edge[1])
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(node_features[u], node_features[v], dim=0).item()

        # Skip edges where cosine similarity is above threshold (i.e., features are "too similar")
        if cos_sim > similarity_threshold:
            rejected_edges += 1  # Increment the counter
            continue
        
        # Otherwise, add the edge
        G.add_edge(u, v)

    print('Number of rejected edges: ',rejected_edges)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Calculate edge weights (cosine similarities)
    calculate_edge_weights(G, node_features)

    # Create data_x and data_y directly using the edges of the graph
    edges = list(G.edges())
    data_x = torch.zeros((len(edges), 2 * node_features.shape[1]), dtype=torch.float32)
    data_y = torch.zeros(len(edges), dtype=torch.float32)
    
    print(f"Processing {len(edges)} edges...")
    
    # Process edges in batches to save memory
    batch_size = 10000
    for i in range(0, len(edges), batch_size):
        batch_edges = edges[i:i + batch_size]
        for j, (u, v) in enumerate(batch_edges):
            idx = i + j
            data_x[idx] = torch.cat([node_features[u], node_features[v]], dim=0)
            data_y[idx] = torch.tensor(G[u][v]['weight'], dtype=torch.float32)
        
        if (i + batch_size) % 50000 == 0:
            print(f"Processed {i + batch_size} edges...")
    
    # Save the processed data
    print("Saving processed data...")
    torch.save(data_x, data_x_path)
    torch.save(data_y, data_y_path)

    # Save the graph using pickle
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

    print("Data saved successfully!")
    
    return data_x, data_y, G

if __name__ == "__main__":
    SEED = 1
    print("Loading ogbn-arxiv dataset...")
    data_x, data_y, G = load_ogbn_arxiv_data(seed=SEED)

    print('Number of training samples: ', data_x.shape[0])
    
    # Print basic information about the loaded data
    print("\nDataset Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Feature dimension: {data_x.shape[1] // 2}")
    print(f"Is graph directed: {nx.is_directed(G)}")
    
    print("\nEdge Weight Statistics:")
    print(f"Mean weight: {data_y.mean().item():.4f}")
    print(f"Min weight: {data_y.min().item():.4f}")
    print(f"Max weight: {data_y.max().item():.4f}")
    
    print("\nGraph Connectivity:")
    print(f"Is graph connected: {nx.is_weakly_connected(G)}")
    print(f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}")
    
    print("\nMemory Usage:")
    print(f"data_x size: {data_x.element_size() * data_x.nelement() / (1024*1024):.2f} MB")
    print(f"data_y size: {data_y.element_size() * data_y.nelement() / (1024*1024):.2f} MB")