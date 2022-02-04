#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import numpy as np


"""
    def forward(self, x):
        N=x.shape[0]
        C=x.shape[1]

        x=x.view(N,C, -1) @ self.Z
        x=x.view(N,C, self.time_dim, self.joints_dim)

        return x
"""


class ConvTemporalSpectral(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 time_dim,
                 joints_dim, in_channels
                 ):
        super(ConvTemporalSpectral, self).__init__()
        self.time_dim = time_dim
        self.joints_dim = joints_dim
        self.Z = nn.Parameter(torch.FloatTensor(joints_dim * time_dim, joints_dim * time_dim))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_block=GNNblock(in_channels,joints_dim * time_dim,device,0.3,squash=False)
        #newGNN(3,joints_dim * time_dim,64,,0.3,4)
        # self.Z = nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(time_dim)
        self.Z.data.uniform_(-stdv, stdv)

    def forward(self, x):
        N = x.shape[0]
        C = x.shape[1]
        x = x.view(N, C, -1)
        x=x.permute(0,2,1)
        x=self.gnn_block(x)
        x = x.permute(0, 2, 1)
        #x = x.view(N, C, -1) @ self.Z
        x = x.view(N, C, self.time_dim, self.joints_dim)

        return x


def create_mlp(dim_input, dim_output, arch, activation = nn.ReLU):
# arch is a list of int

    if len(arch) > 0:
        modules = [nn.Linear(dim_input, arch[0]), activation()]
    else:
        modules = []

    for idx in range(len(arch) - 1):
        modules.append(nn.Linear(arch[idx], arch[idx + 1]))
        modules.append(activation())

    if dim_output > 0:
        last_layer_dim = arch[-1] if len(arch) > 0 else dim_input
        modules.append(nn.Linear(last_layer_dim, dim_output))

    return modules

class newGNN(nn.Module):
    def __init__(self, in_feat, feat_dim, hidden_feat, device,  dropout, num_blocks=10):
        super(newGNN, self).__init__()
        self.in_feat=in_feat
        self.feat_dim=feat_dim
        self.hidden_feat=hidden_feat
        self.num_blocks=num_blocks
        self.gnn_blocks=list()
        self._device=device

        #self.mlp=create_mlp(self.in_feat,self.in_feat,[])
        #self.mlp=nn.Sequential(*self.mlp)
        self.gc1 = GraphConvolution(self.in_feat, self.hidden_feat, self.feat_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_feat*self.feat_dim)
        self.do_prob = dropout

        for i in range(0, self.num_blocks):
            self.gnn_blocks.append(GNNblock(self.hidden_feat, self.feat_dim, self._device, do_prob=self.do_prob))

        self.gnn_blocks = nn.ModuleList(self.gnn_blocks)

        self.gc_final = GraphConvolution(self.hidden_feat, self.in_feat, self.feat_dim)

        self.activation = nn.PReLU()



        self.dropout = nn.Dropout(self.do_prob)

    def forward(self, input):

        gnn_block_output = self.gc1(input)
        b, n, f = gnn_block_output.shape
        gnn_block_output = self.bn1(gnn_block_output.view(b, -1)).view(b, n, f)
        gnn_block_output = self.activation(gnn_block_output)
        out = self.dropout(gnn_block_output)

        for i in range(0,self.num_blocks):
            gnn_block_output=self.gnn_blocks[i](gnn_block_output)

        gnn_block_output=self.gc_final(gnn_block_output)
        return gnn_block_output+input

class GraphConvolution(nn.Module):

    def __init__(self, in_feat, out_feat, feat_dim):
        """
        Perform required matrix multiplications. Adjacency and Weight matrices are learnable.
        :param in_feat: second dim of input matrix H_i elem_of R^(KxF)
        :param out_feat: second dim of output matrix H_i+1 elem_of R^(KxF_hat)
        :param feat_dim: first dim of input and output matrices, K
        """
        super(GraphConvolution, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.adj = Parameter(torch.FloatTensor(self.feat_dim, self.feat_dim))
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.adj.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj, output)
        return output

class GNNblock(nn.Module):
    def __init__(self, in_feat, feat_dim, device, do_prob, squash=True):
        super(GNNblock, self).__init__()
        self.in_feat=in_feat
        self.feat_dim=feat_dim
        self._device=device
        self.do_prob=do_prob
        self.squash=squash
        self.Scatter1= ScatteringLayer(in_feat, feat_dim, device=self._device, num_elements=1, do_prob=self.do_prob)
        self.Scatter2 = ScatteringLayer(in_feat, feat_dim, device=self._device, num_elements=3, do_prob=self.do_prob)
        self.SpectralAtt=SpectralAttention(in_feat, feat_dim,3, self._device)


    def forward(self, input):

        #with record_function("model_gnn_scatter"):
        outputs=self.Scatter1(input.unsqueeze(1), bn=True, dropout=True, squash=True)
        #outputs=self.Scatter2(outputs, bn=True, dropout=True, squash=True)
        #outputs=torch.cat(outputs, dim=1)
        # with record_function("model_gnn_spectral_attention"):
        att_coeffs = self.SpectralAtt(input,outputs).view(-1,3*self.feat_dim).unsqueeze(1).expand(-1,self.in_feat,-1)
        #outputs=torch.kron(input,att_coeffs)
        outputs = outputs.permute(0,3,1,2)
        outputs=outputs.reshape((-1,self.in_feat,3*self.feat_dim))
        outputs=torch.mul(outputs, att_coeffs)
        outputs = outputs.reshape((-1, self.in_feat, 3,self.feat_dim))
        outputs = outputs.permute(0, 2, 3, 1)
        outputs=torch.mean(outputs,dim=1)
        #outputs=outputs.view(-1,self.feat_dim, self.in_feat)

        return outputs+input


class ScatteringLayer(nn.Module):

    def __init__(self, in_feat, feat_dim, num_elements, device, do_prob):
        super(ScatteringLayer, self).__init__()
        self.in_feat=in_feat
        self.feat_dim=feat_dim
        self._device=device
        self.num_elements=num_elements
        self.do_prob=do_prob
        self.adj = Parameter(torch.FloatTensor(self.feat_dim, self.feat_dim))
        self.relu=torch.nn.ReLU()
        N = self.feat_dim
        self.I_cached=torch.eye(N).to(self._device)

        self.bn1 = nn.BatchNorm1d(self.num_elements*self.feat_dim*self.in_feat)
        self.bn2 = nn.BatchNorm1d(self.num_elements * self.feat_dim * self.in_feat)
        self.bn3 = nn.BatchNorm1d(self.num_elements * self.feat_dim * self.in_feat)

        self.dropout = nn.Dropout(self.do_prob)

        self.activation= nn.PReLU()

        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.feat_dim)
        self.adj.data.uniform_(-stdv, stdv)

    def compute_wavelets_torch(self, A):
        N = self.feat_dim # number of nodes

        I=self.I_cached

        # lazy diffusion
        #P = 1 / 2 * (I + A)

        P = 1 / 2 * (I + A / (torch.linalg.matrix_norm(A) ** 2))

        H = (I - P).reshape(1,N, N)

        H_2 = (P - torch.matrix_power(P, 2)).reshape(1,N, N)

        return H,H_2

    def forward(self,input, bn=False, dropout=False, squash=False):
        N = self.feat_dim
        adj=self.adj
        adj=adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj=self.relu(adj)
        wavelet1, wavelet2=self.compute_wavelets_torch(adj)
        input_reshaped=input.view(-1,self.feat_dim,self.in_feat)
        out1=input_reshaped
        out1=torch.matmul(self.adj.reshape(1, N, N), out1)
        if bn:
            out1 = out1.view((-1, self.num_elements* self.feat_dim* self.in_feat))
            out1=self.bn1(out1)

        out1=out1.view((-1,self.num_elements,self.feat_dim,self.in_feat))
        out2=input_reshaped
        out2 = torch.matmul(wavelet1, out2)
        if bn:
            out2 = out2.view((-1, self.num_elements* self.feat_dim* self.in_feat))
            out2=self.bn2(out2)

        out2=out2.view((-1,self.num_elements,self.feat_dim,self.in_feat))
        out3=input_reshaped
        out3 = torch.matmul(wavelet2, out3)
        if bn:
            out3 = out3.view((-1, self.num_elements* self.feat_dim* self.in_feat))
            out3=self.bn3(out3)
        out3=out3.view((-1,self.num_elements,self.feat_dim,self.in_feat))

        concat=torch.cat([out1,out2,out3], dim=1)
        if squash:
            concat=self.activation(concat)

        if dropout:
            concat = self.dropout(concat)

        return concat


    def compute_wavelets_numpy(self, A):

        #computes wavelet filters

        N=A.shape[0] # number of nodes

        I = np.eye(N)

        # lazy diffusion
        #P = 1/2*(I + A/)

        # non-lazy diffusion
        P = 1 / 2 * (I + A / (torch.linalg.matrix_norm(A)**2))

        H = (I - P).reshape(1, N, N)

        H_2 = P-np.linalg.matrix_power(P, 2)

        H_combined = np.concatenate((H, H_2.reshape(1, N, N)), axis=0)

        return H_combined

class SpectralAttention(nn.Module):
    def __init__(self, in_feat, feat_dim, num_freqs, device):
        super(SpectralAttention, self).__init__()
        self.in_feat = in_feat
        self.feat_dim = feat_dim
        self.num_freqs=num_freqs
        self._device=device
        #self.weight_spectral = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.weight_attention = Parameter(torch.FloatTensor(2*self.in_feat))
        #self.channel_mlp=create_mlp(dim_input=self.in_feat*2*self.feat_dim,dim_output=self.in_feat, arch=[])
        #self.channel_mlp = nn.Sequential(*self.channel_mlp)
        #self.ReLU=nn.ReLU()
        self.prelu =  nn.PReLU()
        self.softmax=nn.Softmax(dim=1)
        self.num_scores = 3
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(2*self.in_feat)
        #self.weight_spectral.data.uniform_(-stdv, stdv)
        self.weight_attention.data.uniform_(-stdv, stdv)

    def compute_H_sp(self, input):
        out = torch.matmul(input, self.weight_spectral)
        out=torch.mean(out,dim=1)
        return self.ReLU(out)

    def forward(self,input, scattered):
        output = torch.cat([input.unsqueeze(1).expand(-1, self.num_scores, -1, -1), scattered], dim=3)
        output = torch.matmul(output, self.weight_attention)
        output = self.prelu(output)
        output = self.softmax(output)
        return output


#x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
#return x.contiguous()

# In[3]:


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.fcn=nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            )

        self.gcn = ConvTemporalSpectral(time_dim, joints_dim, out_channels)  # the convolution layer

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )


        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x=self.fcn(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


# In[4]:


class CNN_layer(
    nn.Module):  # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
        (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            , nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


# In[11]:


class Model(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 bias=True):

        super(Model, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 64, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64, 32, [1, 1], 1, input_time_frame,  # 3,3
                                           joints_to_consider, st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(32, 64, [1, 1], 1, input_time_frame,  # 3,3
                                           joints_to_consider, st_gcnn_dropout))

        self.st_gcnns.append(ST_GCNN_layer(64, input_channels, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout))

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size,
                                     txc_dropout))  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for gcn in (self.st_gcnns):
            x = gcn(x)

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x