# Import
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

def generate_data_final_dag(total_nodes, p=0.9, distance_type='type1', seed=None):
    def generate_dag(total_nodes, p):
        """
        Generate a directed acyclic graph (DAG) with the specified number of nodes and edge probability.
        """
        G = nx.gnp_random_graph(total_nodes, p, directed=True)
        DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
        
        assert nx.is_directed_acyclic_graph(DAG), "Generated graph is not a DAG"
        return DAG

    def calculate_distance_matrix(DAG, pos, distance_type):
        """
        Calculate the distance matrix for a DAG where the distance is the Euclidean distance between node coordinates.
        """
        n = len(DAG.nodes)
        distance_matrix = np.zeros((n, n), dtype=float)
        
        for u, v in DAG.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if distance_type == 'type1':
                euclidean = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)
                dag_distance = (euclidean**0.5)*np.log(1+euclidean)**0.5
            elif distance_type == 'type2':
                euclidean = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)
                dag_distance = (euclidean**0.1)*np.log(1+euclidean)**0.9
            elif distance_type == 'type3':
                euclidean = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)
                dag_distance = 1 - 1/(1+euclidean**0.5)   
            elif distance_type == 'type4':
                euclidean = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)
                dag_distance = 1 - np.exp((1-euclidean)/np.log(euclidean))
            elif distance_type == 'type5':
                euclidean = ((x2 - x1) ** 2 + (y2 - y1) ** 2)**(0.5)
                dag_distance = 1 - 1/(1+euclidean**0.2+euclidean**0.5)

            distance_matrix[u, v] = dag_distance
            DAG[u][v]['weight'] = dag_distance
        
        return distance_matrix

    def generate_dag_dataset(total_nodes, p, distance_type):
        """
        Generate the dataset for the DAG, including node coordinates and distance matrix.
        """
        # Generate a DAG
        DAG = generate_dag(total_nodes, p)

        # Generate node positions using graphviz_layout
        pos = nx.drawing.nx_pydot.graphviz_layout(DAG, prog='dot')

        # Normalize the coordinates
        x_coords = np.array([pos[node][0] for node in DAG.nodes()])
        y_coords = np.array([pos[node][1] for node in DAG.nodes()])

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_coords = (x_coords - x_min) / (x_max - x_min)
        y_coords = (y_coords - y_min) / (y_max - y_min)

        pos = {node: (x_coords[i], y_coords[i]) for i, node in enumerate(DAG.nodes())}

        # Calculate the distance matrix
        distance_matrix = calculate_distance_matrix(DAG, pos, distance_type)

        return (
            torch.tensor(distance_matrix, dtype=torch.float32),
            torch.tensor(list(zip(x_coords, y_coords)), dtype=torch.float32),
            DAG,
            pos
        )

    def plot_dag(DAG, pos):
        """
        Plot the generated DAG with node positions and edge weights.
        """
        plt.figure(figsize=(10, 8))
        nx.draw(DAG, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', edge_color='gray', arrows=True)
        edge_labels = nx.get_edge_attributes(DAG, 'weight')
        formatted_labels = {(u, v): f'{w:.2f}' for (u, v), w in edge_labels.items()}
        nx.draw_networkx_edge_labels(DAG, pos, edge_labels=formatted_labels, font_color='red')
        plt.title("Directed Acyclic Graph (DAG)")
        plt.show()

    # Set seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    # Generate the DAG dataset
    distance_matrix, xy_coords, DAG, pos = generate_dag_dataset(total_nodes, p, distance_type)

    # Expand coordinates to create all combinations of node pairs
    n = xy_coords.shape[0]
    expanded_coords = xy_coords.unsqueeze(1).expand(n, n, 2)  # Duplicate xy_coords along new dimensions

    # Create all combinations of xy_coords pairs
    data_x = torch.cat((expanded_coords, expanded_coords.transpose(0, 1)), dim=2).view(-1, 4)

    # Create data_y directly from the distance_matrix
    data_y = distance_matrix.view(-1)

    return data_x, data_y, DAG, pos, plot_dag

# Example usage:
# total_nodes = 100
# data_x, data_y, DAG, pos, plot_dag = generate_data_final_dag(total_nodes, seed=42)
