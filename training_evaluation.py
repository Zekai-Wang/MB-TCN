import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score,f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(sublist, val_loader, model, criterion, optimizer, num_epochs, MB_NUM, batch_size):
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for i in range(MB_NUM):
            train_set = TensorDataset(sublist[i][0].clone().detach().to(device), 
                                      torch.tensor(sublist[i][1]).to(device),
                                     sublist[i][2].clone().detach().to(device))
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            for inputs, labels, masks in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs,masks, active_branch = i)
                final_outputs = torch.reshape(torch.mean(torch.stack(outputs), dim=0),(-1,))
                loss = criterion(final_outputs, labels.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)
        
        val_loss = evaluate_model(val_loader, model, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

    
def evaluate_model(data_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    predictions = []
    true_labels = []
    predicted_prob = []

    with torch.no_grad():
        for inputs, labels, masks in data_loader:
            outputs = model(inputs, masks, active_branch = -1)
            final_outputs = torch.reshape(torch.mean(torch.stack(outputs), dim=0),(-1,))
            loss = criterion(final_outputs, labels.float())
            total_loss += loss.item() * inputs.size(0)
            predicted = torch.round(final_outputs)
            prob = final_outputs
            predictions.extend(predicted.cpu().numpy().reshape(-1))
            true_labels.extend(labels.cpu().numpy().reshape(-1))
            predicted_prob.extend(prob.cpu().numpy().reshape(-1))
    
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)

    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predicted_prob)
    prc_score = average_precision_score(true_labels, predicted_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predicted_prob)
    print(f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} - Recall: {recall:.4f} - Precision: {precision:.4f} "
          f"- F1: {f1:.4f} - AUC: {auc:.4f} - PR Curve Score: {prc_score:.4f}")

    return avg_loss

def split_list(input_list, n):
    
    np.random.shuffle(input_list)
    return np.array_split(input_list, n)


def create_subsets(dataset, label, mask, n_subsets, minority_class):

    indices_one = [i for i, data in enumerate(dataset) if label[i] == minority_class]
    indices_zero = [i for i, data in enumerate(dataset) if label[i] != minority_class]
    
    sub_index_zero = split_list(indices_zero, n_subsets)
    
    min_count = len(indices_one)
    
    subsets = []
    for sub_zero in sub_index_zero:

        indices = indices_one + sub_zero.tolist()
        
        s = dataset[indices]
        l = label[indices]
        m = mask[indices]
        
        subsets.append((s,l,m))
    
    return subsets

def val_create(data, label, mask, batch_size = 128):
    val_set = TensorDataset(data.clone().detach().to(device), 
                        torch.tensor(label).to(device),
                           mask.clone().detach().to(device))
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return val_loader

