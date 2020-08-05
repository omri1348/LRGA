import torch
import torch.nn as nn
import numpy as np


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    tmp_ones = torch.ones((V_T.shape[1],1)).to(torch.device('cuda'))
    norm_factor = torch.mm(U,torch.mm(V_T,tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1/norm_factor

def weight_init(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight.data)
        # nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data,0)
    return

class LowRankAttention(nn.Module):
    def __init__(self,k,d,dropout):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d,4*k),nn.ReLU())
        self.activation = nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        U = tmp[:,:self.k]
        V = tmp[:,self.k:2*self.k]
        Z = tmp[:,2*self.k:3*self.k]
        T = tmp[:,3*self.k:]
        V_T = torch.t(V)
        # normalization
        D = joint_normalize2(U, V_T) 
        res = torch.mm(U, torch.mm(V_T, Z))
        res = torch.cat((res*D,T),dim=1)
        return self.dropout(res)