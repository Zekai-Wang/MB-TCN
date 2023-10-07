from data_processing import *
from MBTCN_Module import *
from training_evaluation import *

import sys
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Python Version:", sys.version)
print('Pytorch Version:', torch.__version__)
print('Numpy Version:', np.__version__)
print('Pandas Version:', pd.__version__)
print('Sklearn Version:', sklearn.__version__)
print('===========================================')
print("Num GPUs Available: ",  torch.cuda.device_count())
print('===========================================')
print('Device Information:',torch.cuda.get_device_name(0))

# read data from psv file, two data files from https://physionet.org/content/challenge-2019/1.0.0/
list_A, list_B = load_data_list('training/training/','training_setB/training_setB/')

#create training and test missingness mask
mask_A, mask_B = missing_mask_matrix(list_A, list_B)

#fill nan
list_A, list_B = fill_nan(list_A, list_B)

#split predictors and response from lists
A_predictors, B_predictors, A_label, B_label = split_predictor_response(list_A, list_B) 

#get training sample indice
sepsis_index_A, nonsepsis_index_A = get_index(A_label)
sepsis_index_B, nonsepsis_index_B = get_index(B_label)

#pads sequences to the same length (3D tensor): SHAPE (N, L, C)
A_padded = create_tensor(A_predictors, padding_value=0)
B_padded = create_tensor(B_predictors, padding_value=0)
A_mask_padded = create_tensor(mask_A, padding_value=0)
B_mask_padded = create_tensor(mask_B, padding_value=0)

# reshape (N, C, L)
A_padded = A_padded.reshape(-1, 40, 336)
B_padded = B_padded.reshape(-1, 40, 336)
A_mask_padded = A_mask_padded.reshape(-1, 40, 336)
B_mask_padded = B_mask_padded.reshape(-1, 40, 336)


print("Summary Information:")
print('----------------------'*2)
total = len(sepsis_index_A) + len(nonsepsis_index_A)
print('Training set:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, len(sepsis_index_A), 100 * len(sepsis_index_A) / total))
print('----------------------'*2)
total2 = len(sepsis_index_B) + len(nonsepsis_index_B)
print('Testing set:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total2, len(sepsis_index_B), 100 * len(sepsis_index_B) / total2))



# initialize MB-TCN model

num_inputs = 40
num_channels = [16,16]
n_outputs = 1
n_branches = 10              
kernel_size = 10
dropout = 0.4
batch_size = 128

model = MBTCN(num_inputs = num_inputs, 
              num_channels = num_channels, 
              n_outputs = n_outputs, 
              n_branches = n_branches,                 
              kernel_size = kernel_size, 
              dropout = dropout).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

#create training set
subset_list = create_subsets(A_padded, A_label, A_mask_padded, n_subsets = 10, minority_class = 1)

#create validation set (in this demo, I select training_B from physionet.org as validation set)
val_loader = val_create(B_padded, B_label, B_mask_padded)


# Training the model
num_epochs = 60
train_model(sublist = subset_list, 
            val_loader = val_loader, 
            model = model, 
            criterion = criterion, 
            optimizer = optimizer, 
            num_epochs = num_epochs, 
            MB_NUM = n_branches, 
            batch_size = batch_size)

