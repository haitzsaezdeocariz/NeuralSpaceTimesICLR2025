# Functions to generate webkb datasets

# Import
import networkx as nx
import numpy as np
import torch
import random
from torch_geometric.datasets import WebKB, Planetoid
from torch_geometric.utils import to_networkx

def load_texas_data(seed=None):
    def calculate_cosine_similarity_matrix(G, node_features):
        """
        Calculate the cosine similarity matrix for a directed network using node features.
        """
        n = len(G.nodes)
        similarity_matrix = np.zeros((n, n), dtype=float)
        
        for u, v in G.edges():
            u_features = node_features[u]
            v_features = node_features[v]
            cos_sim = torch.nn.functional.cosine_similarity(u_features, v_features, dim=0).item()
            cos_sim = max(cos_sim, 0.01)
            similarity_matrix[u, v] = cos_sim
            G[u][v]['weight'] = cos_sim
        
        return similarity_matrix

    # Set seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # Load the Texas dataset
    dataset = WebKB(root='/tmp/Texas', name='Texas')
    data = dataset[0]
    
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=False)  # Keep the graph directed

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Get node features
    node_features = data.x

    # Calculate the cosine similarity matrix
    similarity_matrix = calculate_cosine_similarity_matrix(G, node_features)

    # Create data_x and data_y directly using the edges of the graph (to avoid memory overflow)
    edges = list(G.edges())
    data_x = torch.zeros((len(edges), 2 * node_features.shape[1]), dtype=torch.float32)
    data_y = torch.zeros(len(edges), dtype=torch.float32)
    
    for i, (u, v) in enumerate(edges):
        u_features = node_features[u]
        v_features = node_features[v]
        data_x[i] = torch.cat([u_features, v_features], dim=0)
        data_y[i] = torch.tensor(G[u][v]['weight'], dtype=torch.float32)

    return data_x, data_y, G

def load_cornell_data(seed=None):
    def calculate_cosine_similarity_matrix(G, node_features):
        """
        Calculate the cosine similarity matrix for a directed network using node features.
        """
        n = len(G.nodes)
        similarity_matrix = np.zeros((n, n), dtype=float)
        
        for u, v in G.edges():
            u_features = node_features[u]
            v_features = node_features[v]
            cos_sim = torch.nn.functional.cosine_similarity(u_features, v_features, dim=0).item()
            cos_sim = max(cos_sim, 0.01)
            similarity_matrix[u, v] = cos_sim
            G[u][v]['weight'] = cos_sim
        
        return similarity_matrix

    # Set seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # Load the Cornell dataset
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    data = dataset[0]
    
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=False)  # Keep the graph directed

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Get node features
    node_features = data.x

    # Calculate the cosine similarity matrix
    similarity_matrix = calculate_cosine_similarity_matrix(G, node_features)

    # Create data_x and data_y directly using the edges of the graph (to avoid memory overflow)
    edges = list(G.edges())
    data_x = torch.zeros((len(edges), 2 * node_features.shape[1]), dtype=torch.float32)
    data_y = torch.zeros(len(edges), dtype=torch.float32)
    
    for i, (u, v) in enumerate(edges):
        u_features = node_features[u]
        v_features = node_features[v]
        data_x[i] = torch.cat([u_features, v_features], dim=0)
        data_y[i] = torch.tensor(G[u][v]['weight'], dtype=torch.float32)

    return data_x, data_y, G

def load_wisconsin_data(seed=None):
    def calculate_cosine_similarity_matrix(G, node_features):
        """
        Calculate the cosine similarity matrix for a directed network using node features.
        """
        n = len(G.nodes)
        similarity_matrix = np.zeros((n, n), dtype=float)
        
        for u, v in G.edges():
            u_features = node_features[u]
            v_features = node_features[v]
            cos_sim = torch.nn.functional.cosine_similarity(u_features, v_features, dim=0).item()
            cos_sim = max(cos_sim, 0.01)
            similarity_matrix[u, v] = cos_sim
            G[u][v]['weight'] = cos_sim
        
        return similarity_matrix

    # Set seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # Load the Cornell dataset
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
    data = dataset[0]
    
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=False)  # Keep the graph directed

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Get node features
    node_features = data.x

    # Calculate the cosine similarity matrix
    similarity_matrix = calculate_cosine_similarity_matrix(G, node_features)

    # Create data_x and data_y directly using the edges of the graph (to avoid memory overflow)
    edges = list(G.edges())
    data_x = torch.zeros((len(edges), 2 * node_features.shape[1]), dtype=torch.float32)
    data_y = torch.zeros(len(edges), dtype=torch.float32)
    
    for i, (u, v) in enumerate(edges):
        u_features = node_features[u]
        v_features = node_features[v]
        data_x[i] = torch.cat([u_features, v_features], dim=0)
        data_y[i] = torch.tensor(G[u][v]['weight'], dtype=torch.float32)

    return data_x, data_y, G



