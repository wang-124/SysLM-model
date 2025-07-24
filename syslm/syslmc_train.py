#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, average_precision_score
from sklearn.metrics import roc_curve, auc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from syslm.losses import Calculate_losses
from syslm.syslmc_model import CausalBlock
from syslm.evaluation import calculate_metrics

from torch.optim import lr_scheduler



class MyDataset2(Dataset):
    def __init__(self, data, SG=None, SC=None, label=None, lsa_data=None, 
                 flash_health=None, flash_disease=None, dynamic_health=None, dynamic_disease=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.SG = torch.tensor(SG, dtype=torch.float32) if SG is not None else None
        self.SC = torch.tensor(SC, dtype=torch.float32) if SC is not None else None
        self.label = torch.tensor(label, dtype=torch.float32) if label is not None else None
        self.lsa_data = torch.tensor(lsa_data, dtype=torch.float32) if lsa_data is not None else None
        
       
        self.flash_health = torch.tensor(flash_health, dtype=torch.float32) if flash_health is not None else None
        self.flash_disease = torch.tensor(flash_disease, dtype=torch.float32) if flash_disease is not None else None
        

        self.dynamic_health = torch.tensor(dynamic_health, dtype=torch.float32) if dynamic_health is not None else None
        self.dynamic_disease = torch.tensor(dynamic_disease, dtype=torch.float32) if dynamic_disease is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
       
        return (
            self.data[index], 
            self.SG[index] if self.SG is not None else None, 
            self.SC[index] if self.SC is not None else None, 
            self.label[index] if self.label is not None else None,
            self.lsa_data[index] if self.lsa_data is not None else None, 
            self.flash_health,  
            self.flash_disease, 
            self.dynamic_health, 
            self.dynamic_disease 
        )
def collate_fn2(batch):

    data, SG, SC, label, lsa_data, flash_health, flash_disease, dynamic_health, dynamic_disease = zip(*batch)
    
   
    data = torch.stack(data)
    label = torch.stack(label) if label[0] is not None else None
    lsa_data = torch.stack(lsa_data) if lsa_data[0] is not None else None

   
    SG = torch.stack(SG) if SG[0] is not None else None
    SC = torch.stack(SC) if SC[0] is not None else None
    
   
    flash_health = flash_health[0] 
    flash_disease = flash_disease[0]  
    

    dynamic_health = dynamic_health[0] 
    dynamic_disease = dynamic_disease[0]  

    return (data, SG, SC, label, lsa_data, flash_health, flash_disease, dynamic_health, dynamic_disease)


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



def classify_train(model, Epoch_num, Batch_size, learning_rate, train_loader, test_loader, weights):

    train_losses = []
    epochs = []
    train_aucs = []
    train_accs = []
    train_auprs = []
    train_recalls = []
    train_precisions = []
    train_f1s = []    
    val_losses = []
    val_aucs = []
    val_auprs = []
    val_accs = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-5)#
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    model.train()
    best_auc = float('-inf') 
    best_model = None
    for epoch in range(Epoch_num):
        model.train()
        optimizer.zero_grad()  # 梯度初始化为零
        train_loss = 0.0     
        train_auc = 0.0
        train_acc = 0.0
        train_aupr = 0.0
        train_recall = 0.0
        train_precision = 0.0
        train_f1 = 0.0
        for step, (batch_x, batch_SG, batch_SC, batch_label,batch_lsa,batch_ctrl,batch_case,batch_dyctrl,batch_dycase) in enumerate(train_loader):
            train_input = batch_x.to(device)
            y_true = batch_label.to(device)
          
            train_SG = batch_SG.to(device) if batch_SG is not None else None
            train_SC = batch_SC.to(device) if batch_SC is not None else None
            batch_lsa = batch_lsa.to(device)  #
            batch_ctrl = batch_ctrl.to(device)  # 
            batch_case = batch_case.to(device)  # 
            batch_ctrl_dynamic = batch_dyctrl.to(device)  # 
            batch_case_dynamic = batch_dycase.to(device)  # 

            space1, space2, space3, y_pred = model(train_input, train_SG, train_SC, y_true, batch_lsa, batch_ctrl, batch_case,batch_ctrl_dynamic,batch_case_dynamic)
            
            loss =  Calculate_losses(y_true, y_pred, space1, space2, space3, weights)
 
            
            loss.backward()
    
            for module in model.modules():
                if isinstance(module, CausalBlock):
                    module.zero_diagonal_grad()
            optimizer.step()  
            train_loss += loss.item()
            
           
            metrics = calculate_metrics(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            

            tra_auc = metrics["AUC"]
            tra_aupr = metrics["AUPR"]
            tra_acc = metrics["ACC"]
            tra_precision = metrics["Recall"]#, average='binary'
            tra_recall = metrics["Precision"]
            tra_f1 = metrics["F1-Score"]
            
            train_auc += tra_auc
            train_aupr += tra_aupr
            train_acc += tra_acc
            train_precision += tra_precision
            train_recall += tra_recall
            train_f1 += tra_f1

        train_loss /= len(train_loader) 
        scheduler.step()
        
        train_auc /= len(train_loader)
        train_aupr /= len(train_loader)
        train_acc /= len(train_loader)
        train_recall /= len(train_loader)
        train_precision /= len(train_loader)
        train_f1 /= len(train_loader)

        val_loss, val_auc, val_acc, val_aupr, val_recall, val_precision, val_f1 = classify_test(
            model, Batch_size, test_loader,weights)

        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        val_auprs.append(val_aupr)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        val_f1s.append(val_f1)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model.state_dict()


        model.load_state_dict(best_model)
        
 
        train_losses.append(train_loss)
        epochs.append(epoch)
        train_aucs.append(train_auc)
        train_auprs.append(train_aupr)
        train_accs.append(train_acc)
        train_recalls.append(train_recall)
        train_precisions.append(train_precision)
        train_f1s.append(train_f1)
        if epoch % 50 == 0: 
            print(f'Epoch: {epoch}, '
                  f'train_Loss: {train_loss:.5f}, '
                  f'train_AUC: {train_auc:.5f}, '
                  f'train_AUPR: {train_aupr:.5f}, '
                  f'train_ACC: {train_acc:.5f}, '
                  f'train_Precision: {train_precision:.5f},'
                  f'train_ Recall: {train_recall:.5f}, '
                  f'train_F1: {train_f1:.5f}\n ',
                  f'val_Loss: {val_loss:.5f}, '
                  f'val_AUC: {val_auc:.5f}, '
                  f'val_AUPR: {val_aupr:.5f}, '
                  f'val_ACC: {val_acc:.5f}, '
                  f'val_Precision: {val_precision:.5f}, '
                  f'val_Recall: {val_recall:.5f}, '
                  f'val_F1: {val_f1:.5f}')    
   
    model.load_state_dict(best_model)

    if test_loader is not None:
        test_loss, test_auc,test_aupr, test_acc,test_recall,test_precision,test_f1 = classify_test(model, Batch_size, test_loader,weights)
    else:
        test_loss, test_auc,test_aupr, test_acc,test_recall,test_precision,test_f1 = None, None, None, None, None, None, None

    return test_loss, test_auc,test_aupr,test_acc,test_recall,test_precision,test_f1


def classify_test(model, Batch_size, test_loader,weights):  

    model = model.to(device)
    model.eval()
    with torch.no_grad():  
  
        test_auc = 0.0
        test_acc = 0.0
        test_aupr = 0.0
        test_recall = 0.0
        test_precision = 0.0
        test_f1 = 0.0
        for step, (batch_x, batch_SG, batch_SC, batch_label,batch_lsa,batch_ctrl,batch_case,batch_dyctrl,batch_dycase) in enumerate(test_loader):
            test_input = batch_x.to(device)
            y_true = batch_label.to(device)
          
            test_SG = batch_SG.to(device) if batch_SG is not None else None
            test_SC = batch_SC.to(device) if batch_SC is not None else None
            test_lsa = batch_lsa.to(device)      
            test_ctrl = batch_ctrl.to(device)     
            test_case = batch_case.to(device)     
            batch_ctrl_dynamic = batch_dyctrl.to(device)  # 
            batch_case_dynamic = batch_dycase.to(device)  # 

            space1, space2, space3, y_pred = model(test_input, test_SG, test_SC, y_true, test_lsa, test_ctrl, test_case, batch_ctrl_dynamic,batch_case_dynamic)
            
            loss =  Calculate_losses(y_true, y_pred, space1, space2, space3, weights)
       
            test_loss = loss.item()     
    
      
            metrics = calculate_metrics(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            val_auc = metrics["AUC"]
            val_aupr = metrics["AUPR"]
            val_acc = metrics["ACC"]
            val_precision = metrics["Recall"]
            val_recall = metrics["Precision"]
            val_f1 = metrics["F1-Score"]    
    
            test_auc += val_auc
            test_aupr += val_aupr
            test_acc += val_acc
            test_precision += val_precision
            test_recall += val_recall
            test_f1 += val_f1
        
        test_loss /= len(test_loader)        
        test_auc /= len(test_loader)
        test_aupr /= len(test_loader)
        test_acc /= len(test_loader)
        test_recall /= len(test_loader)
        test_precision /= len(test_loader)
        test_f1 /= len(test_loader)
                  
    return test_loss, test_auc, test_aupr, test_acc, test_recall, test_precision, test_f1

