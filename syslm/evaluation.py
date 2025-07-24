#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, average_precision_score
from sklearn.metrics import roc_curve, auc

def calculate_metrics(y_true, y_pred):

    # Flatten the prediction and true values
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()

    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_value = auc(fpr, tpr)
    aupr = average_precision_score(y_true, y_pred)

    # Find the optimal threshold using the ROC curve
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)

    # Calculate other metrics
    acc = accuracy_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    # Return the calculated metrics
    return {
        "AUC": auc_value,
        "AUPR": aupr,
        "ACC": acc,
        "Recall": recall,
        "Precision": precision,
        "F1-Score": f1
    }


# In[ ]:




