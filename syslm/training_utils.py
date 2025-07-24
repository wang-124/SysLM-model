#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import random
from sklearn import preprocessing
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import lr_scheduler
from sklearn.metrics import pairwise_distances

class MyDataset1(Dataset):
    def __init__(self, data1, data2, mask_array, one_hot_encoded_data):
        self.data1 = torch.tensor(data1, dtype=torch.float32)
        self.data2 = torch.tensor(data2, dtype=torch.float32)
        self.mask_array = torch.tensor(mask_array, dtype=torch.bool)  # 
        self.one_hot_encoded_data = torch.tensor(one_hot_encoded_data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data1)
    
    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.mask_array[index],self.one_hot_encoded_data[index]

def collate_fn1(batch):
    data1, data2, mask_array,one_hot_encoded_data = zip(*batch)
    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    mask_array = torch.stack(mask_array)  
    one_hot_encoded_data = torch.stack(one_hot_encoded_data)
    
    return data1, data2, mask_array, one_hot_encoded_data


def shannon_index(samples):

    p = samples + 1e-10  # avoid log(0)
    shannon_entropy = -torch.sum(p * torch.log(p), dim=1)
    shannon_entropy = shannon_entropy.unsqueeze(1)#
    return shannon_entropy


def bray_curtis(samples):
    samples_np = samples.detach().cpu().numpy()  # 
    distances_np = pairwise_distances(samples_np, metric='braycurtis')  #  Bray-Curtis 
    distances = torch.from_numpy(distances_np).to(samples.device)  # 
    return distances


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def Frobenius_loss(A, B):

    loss = torch.norm(A - B, p='fro')

    return loss


# 训练
def Infer_train(model, Epoch_num, Batch_size, learning_rate, train_loader, test_loader,lam1,lam2, device):

    train_losses = []
    train_mses = []
    train_rmses = []
    train_maes = []
    train_r2s = []
    epochs = []
    losses = []
    lr_list = [] 
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)#, weight_decay=1e-4
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    model.train()
    for epoch in range(Epoch_num):

        train_loss = 0.0
        train_mse = 0.0
        train_rmse = 0.0
        train_mae = 0.0    
        train_r2 = 0.0
        
#         confounders
        for step, (batch_x,batch_x_ra,batch_mask,batch_factor) in enumerate(train_loader):
            optimizer.zero_grad() 
            
            train_input = batch_x.to(device)
            train_ra = batch_x_ra.to(device)
            train_mask = batch_mask.to(device)
            train_confounders = batch_factor.to(device)
            y_pred, y_replaced = model(train_input,train_ra, train_mask, train_confounders)#

            y_pred_masked = torch.masked_select(y_pred, train_mask)
            train_input_masked = torch.masked_select(train_ra, train_mask)

            loss1 = F.mse_loss(y_pred_masked, train_input_masked)
      
            otu_num = train_mask.shape[1]
#             print(otu_num)
            train_mask1 = train_mask.permute(0,2,1).reshape(-1, otu_num)  # 
            y_pred_masked1 = y_pred.permute(0,2,1).reshape(-1, otu_num)  # 
            train_input_masked1 = train_ra.permute(0,2,1).reshape(-1, otu_num)  

            first_col_mask = train_mask1[:, 0]

            train_input_d = train_input_masked1[first_col_mask]
            y_pred_d = y_pred_masked1[first_col_mask]
            
            alphad1 = shannon_index(train_input_d)
            alphad2 = shannon_index(y_pred_d)
            betad1 = bray_curtis(train_input_d)

            betad2 = bray_curtis(y_pred_d)
            loss2 = Frobenius_loss(alphad1,alphad2)
            loss3 = Frobenius_loss(betad1, betad2)    

#             lambda1 = loss1 / (loss1+loss2+loss3)
            lambda1 = lam1 #loss2 / (loss1+loss2+loss3)
            lambda2 = lam2 #loss3 / (loss1+loss2+loss3)
            loss = loss1+ lambda1*loss2+lambda2*loss3
#             loss = loss1+ loss2+loss3
            tra_pred = y_pred_masked.detach().cpu().numpy()
            tra_true = train_input_masked.detach().cpu().numpy()
            tra_mse = mean_squared_error(tra_true, tra_pred)
            tra_rmse = rmse(tra_true, tra_pred)
            tra_mae = mean_absolute_error(tra_true, tra_pred)
            tra_r2 = r2_score(tra_true, tra_pred)

            loss.backward()
            
            optimizer.step()          
            train_loss += loss.item()
            train_mse += tra_mse
            train_rmse += tra_rmse
            train_mae += tra_mae
            train_r2 += tra_r2
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        train_mse /= len(train_loader)
        train_rmse /= len(train_loader)
        train_mae /= len(train_loader)
        train_r2 /= len(train_loader)
        
#         val_loss, val_mse, val_rmse, val_mae,val_r2 = Infer_test(model, Batch_size, test_loader,lam1,lam2,device)

        train_losses.append(train_loss)
        train_mses.append(train_mse)
        train_rmses.append(train_rmse)
        train_maes.append(train_mae)
        train_r2s.append(train_r2)
        epochs.append(epoch)
        lr_list.append(optimizer.param_groups[0]['lr'])  
        
        if epoch % 50 == 0: 
            print(f'Epoch: {epoch}, '
                  f'train_Loss: {train_loss:.5f}, '
                  f'train_MSE: {train_mse:.5f}, '
                  f'train_RMSE: {train_rmse:.5f}, '
                  f'train_MAE: {train_mae:.5f}, '
                  f'train_R2: {train_r2:.5f}')
            
#             print(f'Epoch: {epoch}, '
#                   f'val_Loss: {val_loss:.5f}, '
#                   f'val_MSE: {val_mse:.5f}, '
#                   f'val_RMSE: {val_rmse:.5f}, '
#                   f'val_MAE: {val_mae:.5f}, '
#                   f'val_R2: {val_r2:.5f}')

    if test_loader is not None:
        eval_val_loss, eval_val_mse, eval_val_rmse, eval_val_mae,eval_val_r2 = Infer_test(model, Batch_size, test_loader,lam1,lam2,device)
    else:
        eval_val_loss, eval_val_mse, eval_val_rmse, eval_val_mae, eval_val_r2 = None, None, None, None, None

    return eval_val_loss, eval_val_mse, eval_val_rmse, eval_val_mae,eval_val_r2


def Infer_test(model, Batch_size, test_loader,lam1,lam2,device):  #
    model.eval()
    with torch.no_grad():  # 
        test_loss = 0.0
        test_mse = 0.0
        test_rmse = 0.0
        test_mae = 0.0
        test_r2 = 0.0               
        for index, (batch_x, batch_x_ra, batch_mask, batch_factor) in enumerate(test_loader):
            test_input = batch_x.to(device)
            test_ra = batch_x_ra.to(device)
            test_mask = batch_mask.to(device)
            test_confounders = batch_factor.to(device)
            test_pred,test_relaced = model(test_input,test_ra, test_mask, test_confounders)

            test_outputs_masked = torch.masked_select(test_pred, test_mask)
            test_input_masked = torch.masked_select(test_ra, test_mask)


            loss1 = F.mse_loss(test_outputs_masked, test_input_masked) 

      
            otu_num = test_mask.shape[1]
            test_mask1 = test_mask.permute(0,2,1).reshape(-1, otu_num)  
            y_pred_masked1 = test_pred.permute(0,2,1).reshape(-1, otu_num) 
            test_input_masked1 = test_ra.permute(0,2,1).reshape(-1, otu_num)  

            first_col_mask = test_mask1[:, 0]
            # 
            input_d = test_input_masked1[first_col_mask]
            y_pred_d = y_pred_masked1[first_col_mask]

            alphad1 = shannon_index(input_d)
            alphad2 = shannon_index(y_pred_d)
            betad1 = bray_curtis(input_d)
            betad2 = bray_curtis(y_pred_d)  
            loss2 = Frobenius_loss(alphad1,alphad2)
            loss3 = Frobenius_loss(betad1, betad2)  
#             lambda1 = loss1 / (loss1+loss2+loss3)
            lambda2 = lam1 #loss2 / (loss1+loss2+loss3)
            lambda3 = lam2 #loss3 / (loss1+loss2+loss3)
            val_loss = loss1+ lambda2*loss2+lambda3*loss3

            test_pred = test_outputs_masked.detach().cpu().numpy()
            test_true = test_input_masked.detach().cpu().numpy()
            val_mse = mean_squared_error(test_true, test_pred)
            val_rmse = rmse(test_true, test_pred)
            val_mae = mean_absolute_error(test_true, test_pred)
            val_r2 = r2_score(test_true, test_pred)


            test_loss += val_loss.item()
            test_mse += val_mse
            test_rmse += val_rmse
            test_mae += val_mae
            test_r2 += val_r2
   

        test_loss /= len(test_loader)
        test_mse /= len(test_loader)
        test_rmse /= len(test_loader)
        test_mae /= len(test_loader)
        test_r2 /= len(test_loader)
           
    return test_loss, test_mse, test_rmse, test_mae, test_r2


# In[ ]:




