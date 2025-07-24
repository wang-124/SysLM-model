#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F

def dag_loss(A, rho=1, alpha=1):
    d = A.size(0)
 
    A_hadamard = (A * A)/d

    exp_A = torch.matrix_exp(A_hadamard)

    trace = torch.trace(exp_A)

    hw = trace - d

    return (rho / 2) * hw*hw + alpha * hw


def reconstruction_loss(original_vector, reconstructed_vector):

    mse_loss = F.mse_loss(reconstructed_vector, original_vector)
    return mse_loss

def dynamic_loss(C_dynamic):
   
    T = C_dynamic.size(0)
    temporal_loss = 0.0
    for t in range(1, T):
        temporal_loss += torch.norm(C_dynamic[t] - C_dynamic[t-1], p='fro')
    temporal_loss /= (T - 1) 
    return temporal_loss

def consistency_loss(C_dynamic, C_static, F_dynamic, F_static):

  
    C_dynamic_mean = torch.mean(C_dynamic, dim=0)  # (n, n)

    
    causal_consistency_loss = torch.norm(C_dynamic_mean - C_static, p='fro')

   
    F_dynamic_health = F_dynamic[:, -1, :]  
    F_dynamic_disease = F_dynamic[:, -2, :]  

    F_static_health = F_static[-1, :] 
    F_static_disease = F_static[-2, :]  


    F_dynamic_health_mean = torch.mean(
        F_dynamic_health, dim=0)  
    F_dynamic_disease_mean = torch.mean(
        F_dynamic_disease, dim=0) 

    health_consistency_loss = torch.norm(
        F_dynamic_health_mean - F_static_health, p=2)
    disease_consistency_loss = torch.norm(
        F_dynamic_disease_mean - F_static_disease, p=2)
   
    total_loss = causal_consistency_loss + \
        disease_consistency_loss + health_consistency_loss
    return total_loss

def classify_loss(label, pred, alpha=0.25, gamma=2, reduction='mean'):


    loss = F.binary_cross_entropy(pred, label, reduction='none')

    p_t = label * pred + (1 - label) * (1 - pred)
    p_t = torch.clamp(p_t, 1e-7, 1.0)  
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= modulating_factor

    if alpha > 0:
        alpha_factor = label * alpha + (1 - label) * (1 - alpha)
        loss *= alpha_factor

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  
def Calculate_losses(label, prediction, causal_space1, causal_space2, causal_space3, weights):

    Ho1, Hd1, Hc1 = causal_space1
    Ho2, Hd2, Hc2 = causal_space2
    Ho3, Hd3, Hc3 = causal_space3
 
   
    w_rec = weights['w_rec']
    w_dag = weights['w_dag']
    w_dynamic = weights['w_dynamic']
    w_consistency = weights['w_consistency']

    L_cla = classify_loss(label, prediction)

    L_rec = reconstruction_loss(Ho1, Hd1)+reconstruction_loss(Ho2, Hd2)+torch.mean(
        torch.stack([reconstruction_loss(Ho3[t], Hd3[t]) for t in range(Ho3.size(0))])) 

    L_dag = dag_loss(Hc1)+dag_loss(Hc2)+torch.mean(torch.stack(
        [dag_loss(Hc3[t]) for t in range(Hc3.size(0))])) 

    L_dynamic = dynamic_loss(Hc3)

    L_consistency = consistency_loss(Hc3, Hc2, Hd3, Hd2)

    L_losses = L_cla + w_rec*L_rec+w_dag*L_dag+w_dynamic * \
        L_dynamic+w_consistency*L_consistency
    return L_losses

