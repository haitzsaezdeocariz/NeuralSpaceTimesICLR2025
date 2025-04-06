# Imports 

from itertools import tee
import json
import logging
import os
from typing import Generator, Tuple, Dict, Optional, Iterator
import networkx as nx
from tqdm import tqdm
import numpy as np
import torch
import random

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
NODE_DICT = 'node_dict.json'

#https://www.synapse.org/Synapse:syn2787212


def read_data(file_path: str) -> Generator[Tuple, None, None]:
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                items = line.split("\t")
                yield items[0], items[1], items[2]
    else:
        LOGGER.error(f"File not found")


def break_graph_cycles(G: nx.DiGraph) -> nx.DiGraph:
    cycles = nx.simple_cycles(G)
    G_copy = G.copy()
    nr_of_cycles = 0
    for cycle in tqdm(cycles, mininterval=10):
        nr_of_cycles += 1
        try:
            G_copy.remove_edge(cycle[-2], cycle[-1])
        except nx.NetworkXError:
            continue
    LOGGER.info(f"Broke {nr_of_cycles} cycles")
    return G_copy


def make_node_dict(data_iter: Iterator[Tuple]) -> Dict[str, int]:
    node_dict = {}
    idx = 0
    for (i, j, _) in data_iter:
        if i not in node_dict:
            node_dict[i] = idx
            idx += 1
        if j not in node_dict:
            node_dict[j] = idx
            idx += 1
    return node_dict


def write_dict(d: Dict, output_dir: Optional[str]):
    if not output_dir:
        return
    with open(os.path.join(output_dir, NODE_DICT), 'w') as file:
        json.dump(d, file)


def build_graph(
    data_iter: Generator[Tuple, None, None],
    make_dag: bool,
    remove_self_loops: bool = True,
    reverse_directionality: bool = False,
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    G = nx.DiGraph()
    dict_iter, graph_iter = tee(data_iter)
    node_dict = make_node_dict(dict_iter)
    for (x, y, z) in graph_iter:
        if float(z) > 0:
            if remove_self_loops and x == y:
                pass
            elif reverse_directionality:
                G.add_edge(node_dict[y], node_dict[x])
            else:
                G.add_edge(node_dict[x], node_dict[y])
    G = break_graph_cycles(G) if make_dag else G
    return G, node_dict


def get_dream5_graph(
    file_path: str,
    output_dir: Optional[str] = None,
    make_dag: bool = False,
    remove_self_loops: bool = True,
    reverse_directionality: bool = False,
) -> nx.DiGraph:
    dream5_iter = read_data(file_path)
    G, node_dict = build_graph(
        data_iter=dream5_iter,
        make_dag=make_dag,
        remove_self_loops=remove_self_loops,
        reverse_directionality=reverse_directionality,
    )
    write_dict(d=node_dict, output_dir=output_dir)
    return G


# Function to load the DREAM5 data
def load_dream_data(dataset: str, seed=None, output_dir: Optional[str] = None):
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

    if dataset == 'cerevisiae':
        file_path = './raw_data/DREAM5_NetworkInference_GoldStandard_Network4.tsv'
    elif dataset == 'silico':
        file_path = './raw_data/DREAM5_NetworkInference_GoldStandard_Network1.tsv'
    elif dataset == 'coli':
        file_path = './raw_data/DREAM5_NetworkInference_GoldStandard_Network3.tsv'


    # Load the DREAM5 dataset and build the graph
    G = get_dream5_graph(file_path, output_dir=output_dir, make_dag=False, remove_self_loops=True, reverse_directionality=False)
    
    # Convert to NetworkX graph
    G = nx.convert_node_labels_to_integers(G)  # Ensure node labels are integers

    # Generate node positions using graphviz_layout
    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')

    # Normalize the coordinates
    x_coords = np.array([pos[node][0] for node in G.nodes()])
    y_coords = np.array([pos[node][1] for node in G.nodes()])

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x_coords = (x_coords - x_min) / (x_max - x_min)
    y_coords = (y_coords - y_min) / (y_max - y_min)

    node_features = torch.tensor(list(zip(x_coords, y_coords)), dtype=torch.float32)

    # Calculate the cosine similarity matrix
    similarity_matrix = calculate_cosine_similarity_matrix(G, node_features)

    # Create data_x and data_y directly using the edges of the graph
    edges = list(G.edges())
    data_x = torch.zeros((len(edges), 2 * node_features.shape[1]), dtype=torch.float32)
    data_y = torch.zeros(len(edges), dtype=torch.float32)
    
    for i, (u, v) in enumerate(edges):
        u_features = node_features[u]
        v_features = node_features[v]
        data_x[i] = torch.cat([u_features, v_features], dim=0)
        data_y[i] = torch.tensor(G[u][v]['weight'], dtype=torch.float32)

    return data_x, data_y, G