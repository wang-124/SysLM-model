#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from syslm.syslmc_train import MyDataset2, collate_fn2,classify_train,classify_test

from syslm.syslmc_model import SysLMCModel
import numpy as np
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


def run_syslmc_inference(data_store, dataset_name, classification_label, device, loss_weights_map, Batch_size=256, Epoch_num=200, learning_rate=0.001, N_representation=64):
    """
    Run the full SysLMC pipeline.

    Args:
        data_store (dict): Dictionary containing datasets (fulldata, lsa, static, dynamic).
        dataset_name (str): Name of the dataset (e.g., 'BONUS-CF').
        classification_label (str): The classification label (e.g., 'Cysticfibrosis').
        device (torch.device): Torch device ('cpu' or 'cuda').
        loss_weights_map (dict): Dictionary mapping classification labels to loss weight settings.
        batch_size (int, optional): Batch size for training and inference.
        epoch_num (int, optional): Number of epochs for training.
        learning_rate (float, optional): Learning rate for optimizer.
        n_representation (int, optional): Dimension of learned representation.

    Returns:
        tuple: 
            - predictions (numpy array): Predicted labels.
            - causal1_np (numpy array): First causal matrix.
            - causal2_np (numpy array): Second causal matrix.
            - causal3_np (numpy array): Third causal matrix.
    """

    import torch
    from torch.utils.data import DataLoader

    
    full_datas = data_store['fulldata']
    lsa_data = data_store['lsa']
    static_data = data_store['static']
    dynamic_data = data_store['dynamic']

    full_SC = full_datas['full_SC'][:] if 'full_SC' in full_datas else None
    full_SG = full_datas['full_SG'][:] if 'full_SG' in full_datas else None
    full_data = full_datas['full_data'][:] if 'full_data' in full_datas else None
    full_label = full_datas['full_label'][:] if 'full_label' in full_datas else None

    full_lsa = lsa_data['fulldata_lsa']
    full_lsa = np.transpose(full_lsa, (2, 0, 1))

    full_static_health = static_data['healthy_data']
    full_static_disease = static_data['disease_data']

    full_static_health = np.expand_dims(full_static_health, axis=0)
    full_static_disease = np.expand_dims(full_static_disease, axis=0)

    full_dynamic_health = dynamic_data['healthy_data']
    full_dynamic_disease = dynamic_data['disease_data']

    
    full_torch_dataset = MyDataset2(full_data, full_SG, full_SC, full_label, full_lsa,
                                   full_static_health, full_static_disease,
                                   full_dynamic_health, full_dynamic_disease)

    full_loader = DataLoader(dataset=full_torch_dataset, batch_size=Batch_size,
                             shuffle=False, num_workers=0, collate_fn=collate_fn2)


    N_OTU = full_data.shape[2]
    N_gender = full_SG.shape[1] if full_SG is not None else 1
    N_country = full_SC.shape[1] if full_SC is not None else 1
    N_label = full_label.shape[1]

    lstm_params = {'input_size': N_OTU, 'output_size': N_representation, 'num_layers': 1}
    gcn_params = {'input_dim': N_OTU, 'output_dim': N_representation}
    linear_params = {'N_label': N_label, 'N_country': N_country, 'N_gender': N_gender}
    classify_params = {'input_size': N_representation, "hidden_size": N_representation // 2, 'output_size': 1}

    net2 = SysLMCModel(lstm_params, gcn_params, linear_params, classify_params, dropout=0.5)
    net2.to(device)

    # # Select loss weights for training
    if classification_label in loss_weights_map:
        loss_weights_sets = loss_weights_map[classification_label]
    else:
        raise ValueError(f"No loss weights set found for {classification_label}")

    # 5. 训练
    test_loss, test_auc, test_aupr, test_acc, test_recall, test_precision, test_f1 = classify_train(
        net2, Epoch_num, Batch_size, learning_rate, full_loader, full_loader, loss_weights_sets)

    # 6. 推理
    net2.eval()
    predictions = []
    causal1_np = None
    causal2_np = None
    causal3_np = None

    with torch.no_grad():
        for inputs_x, inputs_SG, inputs_SC, inputs_label, inputs_lsa, inputs_ctrl, inputs_case, inputs_dyctrl, inputs_dycase in full_loader:
            inputs_x = inputs_x.to(device)
            inputs_label = inputs_label.to(device)

            inputs_SG = inputs_SG.to(device) if inputs_SG is not None else None
            inputs_SC = inputs_SC.to(device) if inputs_SC is not None else None
            inputs_lsa = inputs_lsa.to(device)
            inputs_ctrl = inputs_ctrl.to(device)
            inputs_case = inputs_case.to(device)
            inputs_dyctrl = inputs_dyctrl.to(device)
            inputs_dycase = inputs_dycase.to(device)

            out_Space1, out_Space2, out_Space3, out_y_pred = net2(
                inputs_x, inputs_SG, inputs_SC, inputs_label,
                inputs_lsa, inputs_ctrl, inputs_case,
                inputs_dyctrl, inputs_dycase)

            _, _, causal1 = out_Space1
            _, _, causal2 = out_Space2
            _, _, causal3 = out_Space3

            predictions.append(out_y_pred.cpu().numpy())

            if causal1_np is None:
                causal1_np = causal1.cpu().numpy()
                causal2_np = causal2.cpu().numpy()
                causal3_np = causal3.cpu().numpy()

    predictions = np.concatenate(predictions, axis=0)
    return predictions, causal1_np, causal2_np, causal3_np

