#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def calculate_dynamic_threshold(causal_matrix, percentile=80):
    """
    Calculate a dynamic threshold based on the distribution of weights in the causal matrix.
    Keeps a percentage of the strongest edges by absolute value.

    Args:
        causal_matrix (np.array): Causal relationship matrix.
        percentile (float): Percentile to determine the threshold (0–100).

    Returns:
        threshold (float): Calculated threshold value.
    """
    nonzero_values = causal_matrix[causal_matrix != 0]
    threshold = np.percentile(np.abs(nonzero_values), percentile)
    return threshold


def feature_causal(causal_matrix, threshold=0.5):
    """
    Convert a causal matrix into a sparse Directed Acyclic Graph (DAG)
    by removing weak edges below the threshold.

    Args:
        causal_matrix (np.array): Causal relationship matrix.
        threshold (float): Threshold for edge pruning.

    Returns:
        G (nx.DiGraph): Sparse DAG.
    """
    sparse_matrix = np.where(np.abs(causal_matrix) >= threshold, causal_matrix, 0)

    n = sparse_matrix.shape[0]
    edges = [(i, j, sparse_matrix[i, j]) for i in range(n) for j in range(n) if sparse_matrix[i, j] != 0]
    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for edge in edges_sorted:
        i, j, weight = edge
        G.add_edge(i, j, weight=weight)

        try:
            nx.find_cycle(G, orientation='original')
            G.remove_edge(i, j)
        except nx.exception.NetworkXNoCycle:
            pass

    return G


def add_node_labels_and_category(G, taxonomy_name, dataset_name='DIABIMMUNE', taxonomy='P'):
    """
    Assign labels and taxonomy categories to nodes in the graph.

    Args:
        G (nx.DiGraph): Directed graph.
        taxonomy_name (pd.DataFrame): OTU information.
        dataset (str): Dataset identifier.
        taxs (str): Taxonomy level.
    """
    subset = taxonomy_name[(taxonomy_name['Dataset'] == dataset_name) & (taxonomy_name['Taxonomy'] == taxonomy)]

    node_labels = {}

    category_color_map = {
        "P": "#FF7F7F",
        "C": "#8DFF8D",
        "O": "#7F7FFF",
        "F": "#FFFF7F",
        "G": "#FF7FFF",
        "Case": "#BFBFBF",
        "Ctrl": "#BFBFBF"
    }

    node_idx = list(G.nodes())

    for i, (_, row) in enumerate(subset.iterrows()):
        if i < len(node_idx):
            node = node_idx[i]
            category = taxonomy
            color = category_color_map.get(category, "#FFFFFF")

            node_labels[node] = row['OTU_order']
            G.nodes[node]['name'] = row['processed_tax']
            G.nodes[node]['category'] = taxonomy
            G.nodes[node]['label'] = row['processed_tax']
            G.nodes[node]['size'] = 20
            G.nodes[node]['color'] = str(color)

    last_node_idx = len(node_idx) - 2
    G.nodes[node_idx[last_node_idx]]['name'] = "Case"
    G.nodes[node_idx[last_node_idx]]['category'] = 'Case'
    G.nodes[node_idx[last_node_idx]]['label'] = "Case"
    G.nodes[node_idx[last_node_idx]]['size'] = 50
    G.nodes[node_idx[last_node_idx]]['color'] = str(category_color_map['Case'])

    G.nodes[node_idx[last_node_idx + 1]]['name'] = "Ctrl"
    G.nodes[node_idx[last_node_idx + 1]]['category'] = 'Ctrl'
    G.nodes[node_idx[last_node_idx + 1]]['label'] = "Ctrl"
    G.nodes[node_idx[last_node_idx + 1]]['size'] = 50
    G.nodes[node_idx[last_node_idx + 1]]['color'] = str(category_color_map['Ctrl'])


def modify_edges(G):
    """
    Convert all edge weights in the graph to absolute values,
    and assign categorical labels and colors based on direction.

    Args:
        G (nx.DiGraph): Directed acyclic graph.
    """
    edge_color_map = {
        -1: "#5FA5FD",  # Negative edge: blue
        1: "#F1087B"    # Positive edge: pink
    }

    for u, v, data in G.edges(data=True):
        data['category'] = 1 if data['weight'] >= 0 else -1
        data['weight'] = abs(data['weight'])
        data['color'] = edge_color_map.get(data['category'], "#FFFFFF")


def visualize_dag(G, taxonomy_name, dataset_name='DIABIMMUNE', taxonomy='P', title="Sparse DAG"):
    """
    Visualize the directed acyclic graph with node names and weights.

    Args:
        G (nx.DiGraph): DAG.
        taxonomy_name (pd.DataFrame): OTU information.
        dataset (str): Dataset name.
        taxs (str): Taxonomy level.
        title (str): Plot title.
    """
    add_node_labels_and_category(G, taxonomy_name, dataset_name, taxonomy)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', arrowsize=20,
            labels=nx.get_node_attributes(G, 'name'))

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.show()


def extract_directly_connected_nodes(G, dataset_name='DIABIMMUNE', classification_label='milk', taxonomy='P'):
    """
    Identify biomarkers that are directly connected to 'Case' or 'Ctrl' nodes.

    Args:
        G (nx.DiGraph): Directed graph.
        dataset_name (str): Dataset identifier.
        classification_label (str): Experimental or group label.
        taxonomy (str): Taxonomic group name.

    Returns:
        pd.DataFrame: Table of directly connected nodes and metadata.
    """
    node_info = []

    case_node = None
    ctrl_node = None

    for node, data in G.nodes(data=True):
        if data.get('name') == 'Case':
            case_node = node
        elif data.get('name') == 'Ctrl':
            ctrl_node = node

    for u, v in G.edges():
        if u == case_node or u == ctrl_node:
            connected_node = v
        elif v == case_node or v == ctrl_node:
            connected_node = u
        else:
            continue

        connected_node_info = G.nodes[connected_node]
        node_info.append({
            'Node': connected_node,
            'tax': connected_node_info.get('name'),
            'Taxonomy': connected_node_info.get('category'),
            'Dataset': dataset_name,
            'Classification Label': classification_label,
            'Taxonomy Group': taxonomy
        })

    df = pd.DataFrame(node_info).drop_duplicates()
    return df

