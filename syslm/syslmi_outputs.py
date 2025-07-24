#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import random
import collections
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from torch.optim import lr_scheduler

from syslm.training_utils import MyDataset1,collate_fn1, Infer_train, Infer_test

from syslm.syslmi_model import SysLMIModel

def run_syslmi_inference(raw_data_clr, raw_data_ra, mask_data, factors,
                         tcn_params, bilstm_params, fc_params1, fc_params2,
                         dropout=0.5, 
                         Epoch_num=200, Batch_size=32, learning_rate=0.001,
                         lam1=1e-5, lam2=1e-5, 
                         time_points=[3,4,5,6,8,10,12,15],
                         device=None):
    """
    Train SysLMI model and return masked imputation output as numpy array.

 
    Returns:
    -------
    masked_outputs_numpy : np.ndarray
        Predicted outputs (only masked positions) after model inference.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    torch.manual_seed(32)
    torch.cuda.manual_seed_all(32)
    np.random.seed(32)
    random.seed(32)

    # Prepare dataset and dataloader
    full_dataset = MyDataset1(raw_data_clr, raw_data_ra, mask_data, factors)
    full_loader = DataLoader(dataset=full_dataset, batch_size=Batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn1)

    # Initialize model
    model = SysLMIModel(tcn_params, bilstm_params, fc_params1, fc_params2, dropout, time_points, device).to(device)

    # Train model
    Infer_train(model, Epoch_num, Batch_size, learning_rate, full_loader, None,
                lam1=lam1, lam2=lam2, device=device)

    # Inference
    model.eval()
    outputs_list = []
    masked_outputs_list = []

    with torch.no_grad():
        for inputs_clr, inputs_ra, input_masks, input_factors in full_loader:
            inputs_clr = inputs_clr.to(device)
            inputs_ra = inputs_ra.to(device)
            input_masks = input_masks.to(device)
            input_factors = input_factors.to(device)

            outputs, masked_outputs = model(inputs_clr, inputs_ra, input_masks, input_factors)
            masked_outputs_list.append(masked_outputs)

    masked_outputs_tensor = torch.cat(masked_outputs_list, dim=0)
    masked_outputs_numpy = masked_outputs_tensor.cpu().numpy()

    return masked_outputs_numpy

