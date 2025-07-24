#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

def calculate_dynamic_threshold(causal_matrix, percentile=50):
    """
    Dynamically calculate a threshold based on weight distribution
    using a given percentile.
    """
    nonzero_values = causal_matrix[causal_matrix != 0]
    threshold = np.percentile(np.abs(nonzero_values), percentile)
    return threshold

def feature_causal(causal_matrix, threshold=0.5):
    """
    Construct a sparse directed acyclic graph (DAG) from the causal matrix
    by removing edges below a specified threshold and avoiding cycles.
    """
    sparse_matrix = np.where(np.abs(causal_matrix) >= threshold, causal_matrix, 0)
    n = sparse_matrix.shape[0]
    edges = [(i, j, sparse_matrix[i, j]) for i in range(n) for j in range(n) if sparse_matrix[i, j] != 0]
    edges_sorted = sorted(edges, key=lambda x: abs(x[2]), reverse=True)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i, j, weight in edges_sorted:
        G.add_edge(i, j, weight=weight)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(i, j)
    return G

def add_node_labels_and_category(G, taxonomy_name, dataset, taxs):
    """
    Add 'name' and 'category' attributes to graph nodes based on taxonomy.
    """
    subset = taxonomy_name[(taxonomy_name['Dataset'] == dataset) & (
        taxonomy_name['Taxonomy'] == taxs)]

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
            category = taxs
            color = category_color_map.get(category, "#FFFFFF")
            G.nodes[node]['name'] = row['processed_tax']
            G.nodes[node]['category'] = taxs
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
    Modify edge weights to absolute values and assign category and color attributes.
    """
    edge_color_map = {
        -1: "#5FA5FD",  # Negative weight
        1: "#F1087B"    # Positive weight
    }

    for u, v, data in G.edges(data=True):
        data['category'] = 1 if data['weight'] >= 0 else -1
        data['weight'] = abs(data['weight'])
        data['color'] = edge_color_map.get(data['category'], "#FFFFFF")

def visualize_dag(G, taxonomy_name, dataset, taxs, title="Sparse DAG"):
    """
    Visualize the DAG with node names, edge weights, and taxonomy categories.
    """
    add_node_labels_and_category(G, taxonomy_name, dataset, taxs)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue',
            arrowsize=20, labels=nx.get_node_attributes(G, 'name'))
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.show()

def extract_directly_connected_nodes(G, dataset_name, classification_label, taxonomy):
    """
    Extract nodes directly connected to the 'Case' node.
    """
    node_info = []
    case_node = None

    for node, data in G.nodes(data=True):
        if data.get('name') == 'Case':
            case_node = node

    for u, v, data in G.edges(data=True):
        if u == case_node or v == case_node:
            connected_node = v if u == case_node else u
            connected_node_info = G.nodes[connected_node]
            node_info.append({
                'Node': connected_node,
                'tax': connected_node_info.get('name'),
                'Taxonomy': connected_node_info.get('category'),
                'Dataset': dataset_name,
                'Classification Label': classification_label,
                'Taxonomy Group': taxonomy,
                'Weight': data.get('weight', 0)
            })

    return pd.DataFrame(node_info)

def process_taxonomy_data(otu_name,causal_matrix3, dataset_name='BONUS-CF', classification_label='Cysticfibrosis',taxonomies=['P', 'C', 'O', 'F', 'G']):
    """
    Main function to process all taxonomies and extract direct connections.
    Returns:
        all_taxonomies_df: All directly connected nodes across time points.
        core_taxonomies_df: Core markers that appear in all time points.
    """
    all_taxonomies_df = pd.DataFrame()
    core_taxonomies_df = pd.DataFrame()

    for taxonomy in taxonomies:
        print(f"Processing taxonomy: {taxonomy}")
        # Replace this with your actual loading method or mock data
        # e.g., causal_matrix3 = np.load(...)
        #continue  # Skip file loading; remove this in actual integration

        otu_names = [f"{taxonomy}_{i}" for i in range(causal_matrix3.shape[1])]
        taxonomy_intersection = None
        merged_df_list = []

        for t in range(causal_matrix3.shape[0]):
            print(f"Processing time step: {t}")
            causal_matrix_t = causal_matrix3[t]
            threshold = calculate_dynamic_threshold(causal_matrix_t)
            sparse_dag = feature_causal(causal_matrix_t, threshold=threshold)
            add_node_labels_and_category(sparse_dag, otu_name, dataset_name, taxonomy)
            modify_edges(sparse_dag)
            directly_connected_df = extract_directly_connected_nodes(
                sparse_dag, dataset_name, classification_label, taxonomy=taxonomy)
            directly_connected_df['Time'] = f"T{t}"

            current_taxonomy_set = set(directly_connected_df['tax'])
            taxonomy_intersection = current_taxonomy_set if taxonomy_intersection is None else taxonomy_intersection & current_taxonomy_set
            merged_df_list.append(directly_connected_df)

        print(f"Intersected Taxonomy for {taxonomy}: {taxonomy_intersection}")

        if taxonomy_intersection:
            filtered_merged_df = pd.concat([
                df[df['tax'].isin(taxonomy_intersection)]
                for df in merged_df_list
            ], ignore_index=True)

            core_taxonomies_df = pd.concat([core_taxonomies_df, filtered_merged_df], ignore_index=True)
            core_taxonomies_df = core_taxonomies_df.drop_duplicates(subset='tax', keep='first').reset_index(drop=True)

        all_taxonomies_df = pd.concat([all_taxonomies_df, pd.concat(merged_df_list, ignore_index=True)], ignore_index=True)


    return all_taxonomies_df, core_taxonomies_df

