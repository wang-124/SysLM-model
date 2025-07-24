#!/usr/bin/env python
# coding: utf-8

# In[1]:


from syslm import load_example

from syslm.preprocessing import load_syslmc_data

import pandas as pd
import torch

base_dir = ''
data_store = load_syslmc_data(load_example(base_dir))

lsa_data = data_store['lsa']
static_data = data_store['static']
dynamic_data = data_store['dynamic']
fulldata = data_store['fulldata']


# In[3]:


import torch
import numpy as np
import random
from syslm.syslmc_outputs import run_syslmc_inference

# 
loss_weights = {
    'Combo_1': {                 # 0.5-1.5，total，
        'w_rec': 1e-4,          
        'w_dag': 1e-1,        
        'w_dynamic': 1e-2,    #
        'w_consistency': 1e-3 # 
    },
    'Combo_2': {                  # 1.5-3
        'w_rec': 1e-3,         # 
        'w_dag': 1e-5,         # 
        'w_dynamic': 1e-1,     # 
        'w_consistency': 1e-2  # 
    },
    'Combo_3': {                   # 3-10，
        'w_rec': 1e-1,          # 
        'w_dag': 1e-5,         # 
        'w_dynamic': 1e-2,     # 
        'w_consistency': 1e-1  # 
    },
    'Combo_4': {                 # <0.5or>10，
        'w_rec': 1e-1,         #
        'w_dag': 1e-4,         # 
        'w_dynamic': 1e-2,     # 
        'w_consistency': 1e-5  # 
    }
}

loss_weights_map = {
    'total': loss_weights['Combo_1'],
    'milk': loss_weights['Combo_2'],#97/40=2.43
    'egg': loss_weights['Combo_3'],#108/29=3.72
    'peanut': loss_weights['Combo_4'],#129/8=16.13    
    'Cysticfibrosis': loss_weights['Combo_4'],#25/193=0.13
    'preterm': loss_weights['Combo_2'],#24/8=3
    'changed': loss_weights['Combo_4'],#24/155=0.154
    'pregnant': loss_weights['Combo_2'],#30/16=1.88
    'cd': loss_weights['Combo_1'],#44/37=1.19
    'uc': loss_weights['Combo_2'],#59/22=2.69
    'nonibd': loss_weights['Combo_2'],#59/22=2.69
    'label': loss_weights['Combo_1'],  #10/13=0.77
}

# Set random seed for reproducibility
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preds, c1, c2, c3 = run_syslmc_inference(data_store, 'BONUS-CF', 'Cysticfibrosis', device, loss_weights_map)


# In[ ]:





# In[4]:


#causal_explain
from syslm.causal_explain import calculate_dynamic_threshold,matrix_to_sparse_dag,visualize_dag,add_node_labels,modify_edges

threshold = calculate_dynamic_threshold(c1)  

sparse_dag = matrix_to_sparse_dag(c1, threshold=threshold)
visualize_dag(sparse_dag, title="Cysticfibrosis")

add_node_labels(sparse_dag)
modify_edges(sparse_dag)


# In[5]:


otu_name = pd.read_excel(load_example("processed_merged_taxonomy.xlsx"),index_col=0)
otu_name


# In[7]:


#differential_and_network_biomarkers 
from syslm.differential_and_network_biomarkers import (
    calculate_dynamic_threshold, feature_causal,
    modify_edges, visualize_dag,add_node_labels_and_category,
    extract_directly_connected_nodes
)

# Step 1: Construct DAG
threshold = calculate_dynamic_threshold(c2)
sparse_dag = feature_causal(c2, threshold=threshold)

# Step 2: Modify edges and visualize
add_node_labels_and_category(sparse_dag, otu_name, dataset_name='BONUS-CF', taxonomy='P')
modify_edges(sparse_dag)

# Step 3: Extract directly connected nodes (differential biomarkers)
df_diff = extract_directly_connected_nodes(sparse_dag, dataset_name='BONUS-CF', classification_label='Cysticfibrosis', taxonomy='P')
df_diff 


# In[8]:


#core_biomarker
from syslm.core_biomarker import process_taxonomy_data


dyn_df, core_df = process_taxonomy_data(otu_name, c3, dataset_name='BONUS-CF', classification_label='Cysticfibrosis',taxonomies='P')


# In[9]:


core_df


# In[10]:


from syslm.dynamic_biomarker import identify_dynamic_biomarkers

volatility_df, trend_df, dynamic_df = identify_dynamic_biomarkers(dyn_df)

print("volatility biomarkers：")
print(volatility_df)

print("trend biomarkers：")
print(trend_df)

print("dynamic biomarkers：")
print(dynamic_df)


# In[ ]:




