#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import torch
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import scipy

print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("SciPy version:", scipy.__version__)


# In[3]:


import os
import pandas as pd
import numpy as np
import torch
import random
from sklearn import preprocessing
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import lr_scheduler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import pandas as pd

def load_data(base_path, file_names):
    """
    Load multiple CSV files from a specified base path and return a dictionary of DataFrames.

    Parameters:
    base_path (str): The base directory path where the CSV files are stored.
    file_names (list): A list of CSV file names to be loaded.

    Returns:
    dict: A dictionary where the keys are file names and the values are DataFrames.
    """
    data = {}
    
    for file_name in file_names:
        
        full_path = os.path.join(base_path, file_name)
        
        data[file_name] = pd.read_csv(full_path, index_col=0)
    
    return data



def Get_raw_mask_data(df1,df2,subject_number,time_length):

    column_groups1 = [list(range(i, i + time_length)) for i in range(0, df1.shape[1], time_length)]
    column_groups2 = [list(range(i, i + time_length)) for i in range(0, df2.shape[1], time_length)]


    small_dataframes1 = [df1.iloc[:, group] for group in column_groups1]
    small_dataframes2 = [df2.iloc[:, group] for group in column_groups2]

    big_df1 = pd.concat(small_dataframes1, axis=1)
    big_df2 = pd.concat(small_dataframes2, axis=1)
    numpy_array1 = big_df1.to_numpy()
    numpy_array2 = big_df2.to_numpy()

    small_arrays1 = np.split(numpy_array1, numpy_array1.shape[1] // time_length, axis=1)
    small_arrays2 = np.split(numpy_array2, numpy_array2.shape[1] // time_length, axis=1)


    rows1 = [row for small_array in small_arrays1 for row in small_array]
    rows2 = [row for small_array in small_arrays2 for row in small_array]


    rows_array1 = np.array(rows1)
    rows_array2 = np.array(rows2)
    
    regression_raw_data_ra1 = rows_array1.reshape((subject_number, df1.shape[0], time_length))
    regression_raw_data_clr2 = rows_array2.reshape((subject_number, df2.shape[0], time_length))

    mask = ~np.isnan(df1.to_numpy())

    mask = [pd.DataFrame(mask).iloc[:, group] for group in column_groups1]

    mask_array = np.array([df1.to_numpy() for df1 in mask])# 


    return regression_raw_data_ra1, regression_raw_data_clr2,mask_array


def Get_input(data_raw,time_points,device):

    data_raw = torch.tensor(data_raw).to(device)

    batchsize = data_raw.shape[0]
    seq_num = data_raw.shape[1]
    seq_len = data_raw.shape[2]
    
   
    
    rows_array = data_raw.reshape((batchsize*seq_num, seq_len))

    time_array = np.array(time_points).reshape((1, 1, seq_len))  
    time_array = np.repeat(time_array, batchsize*seq_num, axis=0)  
    

    data = rows_array.reshape((batchsize, seq_num, 1, seq_len))

    time_array_reshaped = torch.tensor(time_array.reshape((batchsize, seq_num, 1, seq_len))).to(device)
    data_with_time = torch.cat((data, time_array_reshaped), axis=2)


    continuous_time_data_diff = torch.diff(data, axis=3)
    continuous_time_diff = torch.diff(time_array_reshaped, axis=3)

    data_with_data_time_gap = torch.cat((continuous_time_data_diff, continuous_time_diff), axis=2)


    subsequences = create_subsequences(rows_array)  

    data_subsequences = subsequences.reshape((batchsize, seq_num, 6, 3))#6,4,2,8
    
    return data_with_time, data_with_data_time_gap, data_subsequences

def create_subsequences(data, window_size=3, step_size=1):
    subsequences = []
    for sequence in data:

        for i in range(0, len(sequence) - window_size + 1, step_size):
            subsequences.append(sequence[i:i+window_size])
    return torch.stack(subsequences)


# In[ ]:

import os
import h5py

def load_h5_file(path):
    with h5py.File(path, 'r') as f:
        return {key: f[key][()] for key in f.keys()}

def load_syslmc_data(base_dir):
    """

    """
    print("Loading SysLM dataset into memory...")

    hdf5_lsa = os.path.join(base_dir, 'LSA_weights.h5')
    hdf5_flashwave_static = os.path.join(base_dir, 'static_weights.h5')
    hdf5_flashwave_dynamic = os.path.join(base_dir, 'dynamic_weights.h5')
    hdf5_train_test = os.path.join(base_dir, 'fulldata.h5')

    # 
    for file in [hdf5_lsa, hdf5_flashwave_static, hdf5_flashwave_dynamic, hdf5_train_test]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    #
    data = {
        'lsa': load_h5_file(hdf5_lsa),
        'static': load_h5_file(hdf5_flashwave_static),
        'dynamic': load_h5_file(hdf5_flashwave_dynamic),
        'fulldata': load_h5_file(hdf5_train_test)
    }

    print("SysLM dataset loaded successfully.")
    return data


