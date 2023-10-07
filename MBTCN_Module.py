import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
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
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
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
        return self.network(x)
    
    
    
class MultiBranchingLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_branches):
        super(MultiBranchingLayer, self).__init__()
        self.branches = nn.ModuleList([nn.Linear(n_inputs, n_outputs) for _ in range(n_branches)])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, active_branch):
        out = []
        for i, branch in enumerate(self.branches):
            if i == active_branch:
                out.append(self.sigmoid(branch(x)))
            else:
                with torch.no_grad():
                    out.append(self.sigmoid(branch(x)))
        return out

    
class MBTCN(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_channels, 
                 n_outputs, 
                 n_branches,                 
                 kernel_size, 
                 dropout):
        
        super(MBTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, 
                                   num_channels, 
                                   kernel_size=kernel_size, 
                                   dropout=dropout)
        self.mask_tcn = TemporalConvNet(num_inputs, 
                                   [8], 
                                   kernel_size=kernel_size, 
                                   dropout=dropout)    
        self.multi_branch = MultiBranchingLayer(num_channels[-1] + 8, n_outputs, n_branches)
        
    def forward(self, x, mask, active_branch):
        x = self.tcn(x)
        mask_out = self.mask_tcn(mask)
        combined = torch.cat([x, mask_out], dim=1)
        return self.multi_branch(combined[:, :, -1], active_branch)

