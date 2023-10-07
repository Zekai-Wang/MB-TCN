import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import time
import os


# In[ ]:


def load_data_list(train_set, test_set):
    """
        download data and save as list format, each element in list is a dataframe
       
        Args:
        train_set: training dataset, 20,336 patients with 1,790 positive
        test_set:  test dataset, 20,000 patients with 1,142 positive
        
        return: training samples list, test samples list
        
    """
    doc_names = os.listdir(train_set)
    doc_names2 = os.listdir(test_set)
    list_A = list()
    list_B = list()

    for i in doc_names:
        doc_unit = pd.read_csv(train_set + i, delimiter="|")
        list_A.append(doc_unit)
    for i in doc_names2:
        doc_unit = pd.read_csv(test_set + i, delimiter="|")
        list_B.append(doc_unit)
    
    return list_A, list_B



#create missing value mask matrix
def missing_mask_matrix(list1, list2):
    """
    create training and test missingness mask without label column, 
    set 0 for missing value, 1 for non-missing value
       
        Args:
        list1: training samples list
        list2:  test samples list
        
        return:  training mask list, test mask list
        
    """
    
    mask_A = list()
    mask_B = list()
    
    for i in list1:
        m = i.notna().astype('int')
        mask_A.append(m.drop(m.columns[len(m.columns)-1], axis=1))
    for j in list2:
        n = j.notna().astype('int')
        mask_B.append(n.drop(n.columns[len(n.columns)-1], axis=1))
    
    for i in range(len(mask_A)):
        mask_A[i] = np.array(mask_A[i])
    
    for j in range(len(mask_B)):
        mask_B[j] = np.array(mask_B[j])
    
    return mask_A, mask_B


def fill_nan(list1, list2): 
    """
    impute missing value, 
    for each set of missing indices, use the value of one row before(same column). 
    in the case that the missing value is the first row, look one row ahead instead,
    if all missing value in one column, replace missing value with 0
       
        Args:
        list1: training samples list
        list2:  test samples list
        
        return: imputed training samples list, imputed test samples list       
    """
    for i in list1:
        i.fillna(method='ffill', inplace=True)
        i.fillna(method='bfill', inplace=True)
        i.fillna(0, inplace=True)
    for j in list2:
        j.fillna(method='ffill', inplace=True)
        j.fillna(method='bfill', inplace=True)
        j.fillna(0, inplace=True)
        
    return list1, list2



# split predictor and response from list
def split_predictor_response(train_list, test_list):
    """
    split predictor and response from list
       
        Args:
        train_list: training samples list
        test_list:  test samples list
        
        return: A_predictors, B_predictors, A_label_np, B_label_np
        
    """
    
    train_list1 = list()
    test_list1 = list()

    A_predictors = list()
    A_response = list()
    B_predictors = list()
    B_response = list()
    
    # convert into np.array
    for i in range(len(train_list)):
        train_list1.append(np.array(train_list[i]))
        
    for j in range(len(test_list)):
        test_list1.append(np.array(test_list[j]))
    
    # split predictors and label
    for i in train_list1:
        A_predictors.append(i[:, :i.shape[1]-1])
        A_response.append(i[:,-1])
    for j in test_list1:
        B_predictors.append(j[:, :j.shape[1]-1])
        B_response.append(j[:,-1])
    
    # set label for every sample
    A_label = len(A_response)*[0]
    for i in range(len(A_response)):
        if 1 in A_response[i]:
            A_label[i] = 1
        
    B_label = len(B_response)*[0]
    for i in range(len(B_response)):
        if 1 in B_response[i]:
            B_label[i] = 1
            

    A_label_np = np.array(A_label)
    B_label_np = np.array(B_label)
    
    return A_predictors, B_predictors, A_label_np, B_label_np


def get_index(label):
    
    """
    return sepsis and non-sepsis sample indice
       
        Args:
        label: sample label

        
        return: sepsis_index_np, nonsepsis_index_np
        
    """
    
    sepsis_index_list = list()
    nonsepsis_index_list = list()
    for i in range(len(label)):
        if label[i] == 1:
            sepsis_index_list.append(i)
        else:
            nonsepsis_index_list.append(i)
            
    sepsis_index_np = np.array(sepsis_index_list)
    nonsepsis_index_np = np.array(nonsepsis_index_list)
    
    return sepsis_index_np, nonsepsis_index_np



## create 3D tensor
def create_tensor(features, padding_value=0, max_length=None, padding='pre', truncating='pre'):
    """
    Pads sequences to the same length.

    Args:
        features: list of 2D sequences (time x variables)
        padding_value: padding value
        max_length: maximum length of all sequences. 
                    If not provided, sequences will be padded to the 
                    length of the longest individual sequence
        padding: String, 'pre' or 'post', pad either before or after each sequence.
        truncating: String, 'pre' or 'post' remove values from sequences larger than maxlen, 
                    either at the beginning or at the end of the sequences.

    Returns:
        padded_tensor: PyTorch tensor with sequences padded to the same length.
    """
    
    # Convert input features to tensors
    tensor_features = [torch.tensor(f, dtype=torch.float32) for f in features]

    # Truncate sequences that exceed max_length
    if max_length:
        if truncating == 'pre':
            tensor_features = [f[-max_length:] for f in tensor_features]
        else:
            tensor_features = [f[:max_length] for f in tensor_features]

    # Use pad_sequence for easy padding
    if padding == 'pre':
        padded_tensor = pad_sequence(tensor_features, batch_first=True, padding_value=padding_value)
    else:
        tensor_features = [f.flip([0]) for f in tensor_features]  # Reverse for 'post' padding
        padded_tensor = pad_sequence(tensor_features, batch_first=True, padding_value=padding_value)
        padded_tensor = padded_tensor.flip([1])  # Reverse back after padding
    
    return padded_tensor



