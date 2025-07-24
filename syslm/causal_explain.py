#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def calculate_dynamic_threshold(causal_matrix, percentile=40):
    """
    Calculate a dynamic threshold based on the distribution of causal weights.

    Args:
        causal_matrix (np.array): The causal matrix.
        percentile (float): Percentile (0–100) to determine the threshold.

    Returns:
        float: Calculated threshold value.
    """
    nonzero_values = causal_matrix[causal_matrix != 0]
    threshold = np.percentile(np.abs(nonzero_values), percentile)
    return threshold


def matrix_to_sparse_dag(causal_matrix, threshold=0.5):
    """
    Convert a causal matrix into a sparse Directed Acyclic Graph (DAG),
    preserving only the first, second, and last columns.

    Args:
        causal_matrix (np.array): The causal matrix.
        threshold (float): Edge weight threshold; edges below this will be removed.

    Returns:
        nx.DiGraph: A sparse DAG graph.
    """
    n = causal_matrix.shape[0]
    modified_matrix = np.zeros_like(causal_matrix)
    modified_matrix[:, 0] = causal_matrix[:, 0]
    modified_matrix[:, 1] = causal_matrix[:, 1]
    modified_matrix[:, -1] = causal_matrix[:, -1]

    sparse_matrix = np.where(np.abs(modified_matrix) >= threshold, modified_matrix, 0)
    edges = [(i, j, sparse_matrix[i, j]) for i in range(n) for j in range(n) if sparse_matrix[i, j] != 0]
    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i, j, weight in edges_sorted:
        G.add_edge(i, j, weight=weight)
        try:
            nx.find_cycle(G, orientation='original')
            G.remove_edge(i, j)  # Remove edge if it creates a cycle
        except nx.exception.NetworkXNoCycle:
            pass

    return G


def visualize_dag(G, title="Sparse DAG", node_labels=None):
    """
    Visualize a DAG with optional custom node labels.

    Args:
        G (nx.DiGraph): The DAG.
        title (str): Title of the plot.
        node_labels (dict): Mapping of node index to label name.
    """
    if node_labels is None:
        node_labels = {0: "S", 1: "SM", 2: "SG", 3: "SY"}

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue',
            arrowsize=20, labels=node_labels)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.show()


def add_node_labels(G, node_labels=None, color_mapping=None):
    """
    Add custom node labels and colors to the graph.

    Args:
        G (nx.DiGraph): The DAG.
        node_labels (dict): Mapping from node index to node name.
        color_mapping (dict): Mapping from node name to color.
    """
    if node_labels is None:
        node_labels = {0: "S", 1: "SM", 2: "SG", 3: "SY"}
    if color_mapping is None:
        color_mapping = {
            "S": "#FF7045",
            "SM": "#7CB93B",
            "SC": "#FFFF00",
            "SG": "#00C7FF",
            "SR": "#ECFFBF",
            "SY": "#D97DD8"
        }

    for node in G.nodes:
        name = node_labels.get(node, f"Node_{node}")
        G.nodes[node]['name'] = name
        G.nodes[node]['label'] = name
        G.nodes[node]['color'] = color_mapping.get(name, "gray")


def modify_edges(G):
    """
    Modify edge weights to absolute values and assign sign labels.

    Args:
        G (nx.DiGraph): The DAG.
    """
    for u, v, data in G.edges(data=True):
        data['category'] = 1 if data['weight'] >= 0 else -1
        data['weight'] = abs(data['weight'])

