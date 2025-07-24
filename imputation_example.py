#!/usr/bin/env python
# coding: utf-8

# In[1]:


from syslm import load_example
import pandas as pd
import torch

otu_data_clr = pd.read_csv(load_example("otu_phylum_CLR.csv"),index_col=0)
otu_data_clr

otu_data_na = pd.read_csv(load_example("otu_phylum_na.csv"),index_col=0)   
otu_data_na

sg = pd.read_csv(load_example("SG.csv"),index_col=0)   
factors1 = np.array(sg)
factors = torch.from_numpy(factors1)
factors.shape


# In[2]:


from syslm.preprocessing import load_data, Get_raw_mask_data, Get_input

raw_data_ra, raw_data_clr, mask_data = Get_raw_mask_data(otu_data_na, otu_data_clr, subject_number=218, time_length=8)

# Prepare input for model
time_points = [3, 4, 5, 6, 8, 10, 12, 15]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_with_time, data_with_time_diff, data_subseq = Get_input(raw_data_clr, time_points, device)

# Output the final data shapes
print('data_with_time.shape:', data_with_time.shape)
print('data_with_time_diff.shape:', data_with_time_diff.shape)
print('data_subseq.shape:', data_subseq.shape)


# In[4]:


import random
from syslm.syslmi_outputs import run_syslmi_inference
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


torch.manual_seed(32)
torch.cuda.manual_seed_all(32)
np.random.seed(32)
random.seed(32)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Define hyperparameters
Epoch_num = 200
Batch_size = 32
learning_rate = 0.001  # [0.01,0.001,0.0001,0.00001]
time_points = [3,4,5,6,8,10,12,15]

tcn_params = {
    "input_channels1": 2,
    "input_channels2": 2,
    "input_channels3": 3,
    "output_channels": [32, 32, 32]
}

bilstm_params = {
    "input_size1": 32*3*6,
    "input_size2": 2,
    "input_size3": 2,
    "input_size4": 3,
    "hidden_size": 64
}


fc_params1 = {
    "linear_dims": (factors.shape[1], 128)
}

fc_params2 = {
    "linear_dims": (256, len(time_points))
}

masked_outputs = run_syslmi_inference(raw_data_clr, raw_data_ra, mask_data, factors,
                                      tcn_params, bilstm_params, fc_params1, fc_params2)

masked_outputs

print("masked_outputs shape:", masked_outputs.shape)


# In[ ]:




