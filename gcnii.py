import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, in_features, out_features, time_dim,
                 joints_dim, residual=False):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features

        self.in_channels=in_channels

        self.out_features = out_features
        self.residual = residual

        self.Z = nn.Parameter(torch.FloatTensor(joints_dim * time_dim, joints_dim * time_dim))

        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, h0 , lamda, alpha, l):

        N = x.shape[0]
        C = self.in_channels
        T = self.time_dim
        V = self.joints_dim

        theta = math.log(lamda/l+1)

        x = x.view(-1, C, T * V)

        x = x.permute(0, 2, 1)

        hi = self.Z @ x

        support = (1-alpha)*hi+alpha*h0
        r = support
        output = theta*(support @ self.weight)+(1-theta)*r
        if self.residual:
            output = output+x
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, dropout, lamda, alpha):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden))
        self.params1 = list(self.convs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        return layer_inner