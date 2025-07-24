#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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
from syslm.preprocessing import load_data, Get_raw_mask_data, Get_input

from torch.nn.utils import weight_norm

torch.manual_seed(32)
torch.cuda.manual_seed_all(32)
np.random.seed(32)
random.seed(32)
# torch.backends.cudnn.deterministic = True

 
def inverse_clr_transform(y):
    exp_y = torch.exp(y)
    return exp_y / torch.sum(exp_y, dim = 1, keepdim = True)  

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_channels, output_channels, dropout):
        super(TCN, self).__init__()
   
        self.tcn = nn.Sequential(TemporalConvNet(input_channels, output_channels, dropout=dropout),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool1d(6))#6,4,2,8
    def forward(self, x):
        out = self.tcn(x)
        return out
    
class TCN_Combination(nn.Module):
    def __init__(self, input_channels1, input_channels2, input_channels3, output_channels, dropout):
        super(TCN_Combination, self).__init__()
        self.tcn1 = TCN(input_channels1, output_channels, dropout=dropout)
        self.tcn2 = TCN(input_channels2, output_channels, dropout=dropout)
        self.tcn3 = TCN(input_channels3, output_channels, dropout=dropout)
        
    def forward(self, *inputs):
        x1, x2, x3 = inputs
        x1 = x1.reshape(-1, *x1.shape[2:])## 
        x2 = x2.reshape(-1, *x2.shape[2:])#
        x3 = x3.reshape(-1, *x3.shape[2:])
        x3 = x3.permute(0, 2, 1)## 
        tcn_out1 = self.tcn1(x1)
        tcn_out2 = self.tcn2(x2)
        tcn_out3 = self.tcn3(x3)        
        tcn_out = torch.cat((tcn_out1, tcn_out2, tcn_out3), dim=1)  
 
        return tcn_out
     
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2) 
    
    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.layer_norm(out)
        return out, _

class WeightedFusion(nn.Module):
    def __init__(self, input_dim, num_sources):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_sources, requires_grad=True))

    def forward(self, *inputs):
        assert len(inputs) == len(self.weights)
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))

        return weighted_sum / self.weights.sum()   
    
class BiLSTM_Attention_Weighted(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3,input_size4, hidden_size):
        super(BiLSTM_Attention_Weighted, self).__init__()
        self.bilstm1 = BiLSTM(input_size1, hidden_size)
        self.bilstm2 = BiLSTM(input_size2, hidden_size)
        self.bilstm3 = BiLSTM(input_size3, hidden_size)
        self.bilstm4 = BiLSTM(input_size4, hidden_size)
        self.weighted_fusion = WeightedFusion(input_dim=hidden_size*2, num_sources=4)
        
    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1) 
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  
        soft_attn_weights = F.softmax(attn_weights, 1)
   
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights  
        
    def forward(self, *inputs):
        tcn_x, x1, x2, x3 = inputs

        tcn_x = tcn_x.reshape(-1,1,32*3*6)  
        x1 = x1.reshape(-1, *x1.shape[2:])## Output shape: 
        x2 = x2.reshape(-1, *x2.shape[2:])## Output shape: 
        x3 = x3.reshape(-1, *x3.shape[2:])## Output shape:   
        x1 = x1.permute(0, 2, 1)## Output shape:
        x2 = x2.permute(0, 2, 1)##      
        lstm_tcn_out, (final_hidden_state, final_cell_state) = self.bilstm1(tcn_x) 
        lstm_out_x1, (final_hidden_state1, final_cell_state1) = self.bilstm2(x1) 
        lstm_out_x2, (final_hidden_state2, final_cell_state2) = self.bilstm3(x2) 
        lstm_out_x3, (final_hidden_state3, final_cell_state3) = self.bilstm4(x3) 

        attn_tcn, attention = self.attention_net(lstm_tcn_out, final_hidden_state)
        attn_lstm1, attention1 = self.attention_net(lstm_out_x1, final_hidden_state1)
        attn_lstm2, attention2 = self.attention_net(lstm_out_x2, final_hidden_state2)
        attn_lstm3, attention3 = self.attention_net(lstm_out_x3, final_hidden_state3)        

        weighted_attn_out = self.weighted_fusion(attn_tcn, attn_lstm1, attn_lstm2, attn_lstm3)
    
        return weighted_attn_out
class SysLMIModel(nn.Module):
    def __init__(self,tcn_params, bilstm_params,fc_params1, fc_params2, dropout,time_points, device):
        super(SysLMIModel, self).__init__()
        self.TCN_model = TCN_Combination(**tcn_params,dropout=dropout)
        self.BiLSTM_model = BiLSTM_Attention_Weighted(**bilstm_params)
        self.confounder_fc1 = nn.Sequential(
            nn.Linear(*fc_params1["linear_dims"]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc_regression = nn.Sequential(
            nn.Linear(*fc_params2["linear_dims"])
#             nn.Sigmoid()
        )
        self.time_points = time_points
        self.device = device

    def forward(self, *inputs):
        x, x_ra, mask,confounders = inputs

        batch_size, seq_num, seq_len = x.size()
        x1,x2,x3 = Get_input(x,self.time_points,self.device)        
      
        tcn_out = self.TCN_model(x1,x2,x3)     
        lstm_out = self.BiLSTM_model(tcn_out,x1,x2,x3)
        lstm_out = lstm_out.view(batch_size, seq_num, -1)#
        conf_embed = self.confounder_fc1(confounders)
        conf_embed = conf_embed.unsqueeze(1).expand(-1, seq_num, -1)#
        reg_input = torch.cat((lstm_out,conf_embed),dim=2)#
        regression_out = self.fc_regression(reg_input) 
        regression_out = inverse_clr_transform(regression_out)

        regression_masked = torch.where(mask, x_ra, regression_out) #替换RA      


        return regression_out, regression_masked

