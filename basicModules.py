from torch import nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)
        if self.outBn:

            if len(x.shape) == 2:
                x = self.bns(x)
            else:
                x = self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct:
            x = self.actFunc(x)
        if self.outDp:
            x = self.dropout(x)
        return x


class LayerAtt2(nn.Module):
    def __init__(self, inSize, outSize):
        super().__init__()
        self.Q = nn.Linear(inSize, outSize)
        self.K = nn.Linear(inSize, outSize)
        self.V = nn.Linear(inSize, outSize)
        self.norm = 1. / (outSize ** 0.5)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        att = torch.matmul(Q, K.transpose(1, 2)) * self.norm
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, V)
        return out.mean(dim=1)
