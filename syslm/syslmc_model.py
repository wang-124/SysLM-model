#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
# from syslmc_train import create_graph_data,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch_geometric.data import Data


def create_graph_data(weight_matrix):
    if weight_matrix.dim() != 3:
        raise ValueError("(num_graphs, num_nodes, num_nodes)")

    num_graphs = weight_matrix.shape[0]
    data_list = []

    for i in range(num_graphs):
        edge_index = torch.nonzero(weight_matrix[i] != 0).T

        edge_attr = weight_matrix[i][edge_index[0], edge_index[1]]

        x = weight_matrix[i]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)

    return data_list


def clr_transform(X):
    X = X+1
    
    geometric_mean = X.prod(dim=1)**(1.0/X.size(1))

    geometric_mean = geometric_mean.unsqueeze(1)
  
    ratio = X / geometric_mean

    log_ratio = torch.log(ratio)
    return log_ratio


class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, output_size//2,
                            num_layers, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(output_size)
        self.fc = nn.Linear(output_size, output_size)

    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        hidden = final_state.view(batch_size, -1, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(
            2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, x):
        out, (h, c) = self.lstm(x) 
        out = self.layer_norm(out)
        attn, _ = self.attention_net(out, h)
        out = self.fc(attn)  
        return out


class GCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GCNModel, self).__init__()
        self.conv1 = gnn.GCNConv(input_dim, output_dim//4)
        self.conv2 = gnn.GCNConv(output_dim//4, output_dim//2)
        self.dropout = nn.Dropout(dropout_rate)  


        self.fc = nn.Linear(output_dim//2, output_dim)  

    def forward(self, data_list):

        batch = Batch.from_data_list(data_list)

        x, edge_index = batch.x, batch.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)


        x = gnn.global_mean_pool(x, batch.batch)


        x = self.fc(x)
        x = torch.relu(x)

        return x  



class LinearTransform(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class SharedEncoder(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(SharedEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            #             nn.LayerNorm(input_size//2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),   
            nn.Linear(input_size//2, input_size//4),
            #             nn.LayerNorm(input_size//4),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate) 
        )

    def forward(self, x):
        return self.mlp(x)


class SharedDecoder(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(SharedDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size//4, input_size//4),
            #             nn.LayerNorm(input_size//4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  
            nn.Linear(input_size//4, input_size//2),
            #             nn.LayerNorm(input_size//2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  
            nn.Linear(input_size//2, input_size),
            #             nn.LayerNorm(input_size),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.mlp(x)


class LabelDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LabelDecoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size+hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = F.relu(self.linear1(x)) 
        out = self.dropout(out) 
        out = torch.cat((x, out), dim=1)  
        out = F.sigmoid(self.linear2(out))  
        return out

class TimeCausalMatrix(nn.Module):
    def __init__(self, size):
        super(TimeCausalMatrix, self).__init__()
        self.timecausal_mat = nn.Parameter(torch.randn(size, size))
        
class CausalBlock(nn.Module):
    def __init__(self, causal_size, linear_size, block_type,iftime):
        super(CausalBlock, self).__init__()
        self.causal_size = causal_size
        self.linear_size = linear_size
        self.block_type = block_type
        self.iftime = iftime
        self.causal_mat = nn.Parameter(torch.randn(
            causal_size, causal_size))  
                    
        if self.iftime == True:
            self.causal_time = nn.ModuleList([TimeCausalMatrix(causal_size) for _ in range(10)])
        
        if self.block_type == 'semantic':
            self.mask = torch.ones(causal_size, causal_size) - \
                torch.eye(causal_size)  
        elif self.block_type == 'feature':
            self.mask = torch.ones(causal_size, causal_size) - \
                torch.eye(causal_size) 
            self.mask[-1, -2] = 0  
            self.mask[-2, -1] = 0  
        self.linear_transforms = nn.ModuleList(
            [nn.Linear(linear_size, linear_size) for _ in range(causal_size)])

    def forward(self, x, causal_size=None,time_steps=None):
              
        causal_size = causal_size if causal_size is not None else self.causal_size
#
        h_prime = []
        for i in range(causal_size):
            transformed = self.linear_transforms[i](x[i])
            h_prime.append(transformed)
       
        h_prime = torch.stack(h_prime, dim=0)  # 【6,32,128】

        causal_mat = self.causal_mat[:causal_size, :causal_size].to(x.device)  
       
        if time_steps is not None and self.iftime == True:
#             print("Time",time_steps)
            causal_mat = causal_mat + self.causal_time[time_steps].timecausal_mat[:causal_size, :causal_size].to(x.device)
        mask = self.mask[:causal_size, :causal_size].to(x.device) 
        
        causal_mat = causal_mat * mask  # 

        h_prime_new = torch.einsum('ik, i... -> k...', causal_mat, h_prime)

        return h_prime_new, causal_mat

    def zero_diagonal_grad(self):
     
        if self.causal_mat.grad is not None:
            with torch.no_grad():

                grad = self.causal_mat.grad
                diag_indices = torch.arange(
                    self.causal_size, device=grad.device)

                if self.block_type == 'semantic':
   
                    grad[diag_indices, diag_indices] = 0

                elif self.block_type == 'feature':
                   
                    grad[-1, -2] = 0  # 
                    grad[-2, -1] = 0  # 

                    grad[diag_indices, diag_indices] = 0 
        else:
      
            print("Warning: causal_mat.grad is None. No gradient to modify.")

class SemSCM(nn.Module):
    def __init__(self, lstm_params, gcn_params, linear_params, clssify_params, dropout):
        super(SemSCM, self).__init__()
        self.dropout = dropout
        self.output_size = lstm_params['output_size'] 
        # input_size, hidden_size, output_size，num_layers
        self.bilstm = BiLSTM(**lstm_params)
        self.gcn = GCNModel(**gcn_params, dropout_rate=self.dropout)
        self.linear1 = LinearTransform(
            linear_params['N_gender'], self.output_size)  # sg
        self.linear2 = LinearTransform(
            linear_params['N_country'], self.output_size)  # sc
        self.lineary = LinearTransform(
            linear_params['N_label'], self.output_size)  # sy
        self.encoder = SharedEncoder(
            input_size=self.output_size, dropout_rate=self.dropout)  
        self.decoder = SharedDecoder(
            input_size=self.output_size, dropout_rate=self.dropout)  

        self.causal_size = 5  
        self.causal = CausalBlock(
            causal_size=self.causal_size, linear_size=self.output_size//4, block_type='semantic',iftime=False)

    def forward(self, *inputs):
        input_sto, input_sg, input_sc, input_sy, input_lsa, _, _, _, _ = inputs
       
        batch_size, time, otu = input_sto.size()

        input_clr = clr_transform(input_sto.reshape(-1, otu))  
        input_clr = input_clr.view(batch_size, time, otu)


        graph_lsa = create_graph_data(input_lsa)

        if input_sg is None or input_sc is None:
            current_causal_size = 4  
        else:
            current_causal_size = self.causal_size  
        

        Hs = self.bilstm(input_clr)  
        Ho = self.gcn(graph_lsa)
        Hg = self.linear1(input_sg) if input_sg is not None else None
        Hc = self.linear2(input_sc) if input_sc is not None else None
        Hy = self.lineary(input_sy)

        H_list = [Hs, Ho, Hc, Hg, Hy]  

       
        H_list = [h for h in H_list if h is not None]

  
        H_original = torch.stack(H_list, dim=0)
        

        H_encoded = self.encoder(H_original)  
        H_causal_rec, H_causal_mat = self.causal(H_encoded, causal_size=current_causal_size)
        H_decoded = self.decoder(H_causal_rec)
        Hy_decoded = H_decoded[-1] 

        return H_original, H_decoded, H_causal_mat, Hy_decoded  # ,label_decoded


class SharedComponents(nn.Module):
    def __init__(self, lstm_params, gcn_params, dropout, iftime=False):
        super(SharedComponents, self).__init__()
        self.output_size = lstm_params['output_size']
        self.gcn = GCNModel(**gcn_params, dropout_rate=dropout)

        self.encoder = SharedEncoder(input_size=self.output_size, dropout_rate=dropout)
        self.decoder = SharedDecoder(input_size=self.output_size, dropout_rate=dropout)
        self.causal = CausalBlock(
            causal_size=8 + 2, linear_size=self.output_size // 4, block_type='feature', iftime=iftime)#8 is N_otu

    def forward(self, input_data, graph_disease, graph_health, time_steps=None):
        # GCN processing
        Hf = input_data
        Hh = self.gcn(graph_health)
        Hd = self.gcn(graph_disease)
        
        # Concatenate
        H_original = torch.cat([Hf, Hd, Hh], dim=0)

        # Encoding and causal inference
        H_encoded = self.encoder(H_original)
        H_causal_rec, H_causal_mat = self.causal(H_encoded)
        H_decoded = self.decoder(H_causal_rec)

        return H_original, H_decoded, H_causal_mat# 二维或者三维


class StaticFeaSCM(nn.Module):
    def __init__(self, lstm_params, gcn_params, linear_params, dropout):
        super(StaticFeaSCM, self).__init__()
        self.dropout = dropout
        self.output_size = lstm_params['output_size']
        self.shared = SharedComponents(lstm_params, gcn_params, dropout, iftime=False)
  
        self.max_size = 3000  
        self.linear1 = nn.Linear(in_features=self.max_size, out_features=self.output_size)
    def forward(self, *inputs):
        input_sto, _, _, _, _, input_ctrl, input_case, _, _ = inputs
        batch_size, time, feature_d = input_sto.size()

      
        merged_x = input_sto.view(-1, feature_d)
        merged_x = merged_x.transpose(0, 1) 

        actual_input_size = min(merged_x.size(1), self.max_size)
        
       
        Hf  = torch.nn.functional.linear(merged_x, self.linear1.weight[:, :actual_input_size], self.linear1.bias)


        # Create graph data for control and case groups
        graph_flash_health = create_graph_data(input_ctrl)
        graph_flash_disease = create_graph_data(input_case)
        
#         print("Shape of graph_flash_health:", len(graph_flash_health),graph_flash_health[0])
#         print("Shape of graph_flash_disease:", len(graph_flash_disease),graph_flash_disease[0])
        
        # Use shared components
        H_original, H_decoded, H_causal_mat = self.shared(Hf, graph_flash_disease, graph_flash_health)

        return H_original, H_decoded, H_causal_mat


class DynamicFeaSCM(nn.Module):
    def __init__(self, lstm_params, gcn_params, linear_params, dropout):
        super(DynamicFeaSCM, self).__init__()
        self.dropout = dropout
        self.output_size = lstm_params['output_size']
        self.shared = SharedComponents(lstm_params, gcn_params, dropout,iftime=True)
        self.max_size = 300  
        self.shared_linear = nn.Linear(in_features=self.max_size, out_features=self.output_size)
        
    def forward(self, *inputs):
        input_sto, _, _, _, _, _, _, input_dyn_ctrl, input_dyn_case = inputs
        batch_size, time, feature_d = input_sto.size()
     

        input_x = input_sto.permute(1, 0, 2).permute(0, 2, 1)

        Hft = torch.nn.functional.linear(input_x, self.shared_linear.weight[:, :batch_size], self.shared_linear.bias)
        # Create graph data for dynamic control and case groups
        graph_flash_health_dyn = create_graph_data(input_dyn_ctrl)
        graph_flash_disease_dyn = create_graph_data(input_dyn_case)
        dynamic_health = self.shared.gcn(graph_flash_health_dyn).unsqueeze(1) 
        dynamic_disease = self.shared.gcn(graph_flash_disease_dyn).unsqueeze(1) 

        # Use shared components
        H_original = torch.cat([Hft, dynamic_disease, dynamic_health], dim=1)

        # Unbind along the time dimension for each time step
        H_original_list = torch.unbind(H_original, dim=0)

        H_decoded_list = []
        H_causal_mat_list = []

        # Process each time step
        for t in range(len(H_original_list)):
            H_encoded = self.shared.encoder(H_original_list[t])
            H_causal_rec, H_causal_mat = self.shared.causal(H_encoded, time_steps=t)#使用时间特异性因果
            H_decoded = self.shared.decoder(H_causal_rec)

            H_decoded_list.append(H_decoded)
            H_causal_mat_list.append(H_causal_mat)

       
        H_decoded_final = torch.stack(H_decoded_list, dim=0)
        H_causal_mat_final = torch.stack(H_causal_mat_list, dim=0)

        return H_original, H_decoded_final, H_causal_mat_final
    
class SysLMCModel(nn.Module):
    def __init__(self, lstm_params, gcn_params, linear_params, classify_params, dropout):
        super(SysLMCModel, self).__init__()
        self.dropout = dropout
        self.semcm = SemSCM(lstm_params, gcn_params,
                            linear_params, classify_params, self.dropout)
        self.static = StaticFeaSCM(
            lstm_params, gcn_params, linear_params, self.dropout)
        self.dynamic = DynamicFeaSCM(
            lstm_params, gcn_params, linear_params, self.dropout)
        self.label_decoder = LabelDecoder(
            **classify_params, dropout_rate=self.dropout)  

    def forward(self, *inputs):
        H_o1, H_d1, H_c1, Hy = self.semcm(*inputs) 
        H_o2, H_d2, H_c2 = self.static(*inputs)
        H_o3, H_d3, H_c3 = self.dynamic(*inputs)
        prediction = self.label_decoder(Hy)
        return (H_o1, H_d1, H_c1), (H_o2, H_d2, H_c2), (H_o3, H_d3, H_c3),prediction


# In[ ]:




