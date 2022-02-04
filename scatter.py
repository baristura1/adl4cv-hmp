import numpy as np
import torch
import torch.nn as nn
import math
from torch.profiler import profile, record_function, ProfilerActivity

from torch.nn.parameter import Parameter

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

        self.activation = nn.Tanh()



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
        self.SpectralAtt=SpectralAttention(in_feat, feat_dim,9, self._device)


    def forward(self, input):

        #with record_function("model_gnn_scatter"):
        outputs=self.Scatter1(input.unsqueeze(1), bn=True, dropout=True, squash=True)
        outputs=self.Scatter2(outputs, bn=True, dropout=True, squash=True)
        #outputs=torch.cat(outputs, dim=1)
        # with record_function("model_gnn_spectral_attention"):
        att_coeffs = self.SpectralAtt(outputs)
        outputs=outputs.view((input.shape[0],9,-1)).transpose(1,2)
        outputs=torch.bmm(outputs, att_coeffs.unsqueeze(2))
        outputs=outputs.view(-1,self.feat_dim, self.in_feat)
        return outputs+input


class ScatteringLayer(nn.Module):

    def __init__(self, in_feat, feat_dim, num_elements, device, do_prob):
        super(ScatteringLayer, self).__init__()
        self.in_feat=in_feat
        self.feat_dim=feat_dim
        self._device=device
        self.num_elements=num_elements
        self.do_prob=do_prob
        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.weight2 = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.weight3 = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.adj = Parameter(torch.FloatTensor(self.feat_dim, self.feat_dim))
        N = self.feat_dim
        self.I_cached=torch.eye(N).to(self._device)

        self.bn1 = nn.BatchNorm1d(self.num_elements*self.feat_dim*self.in_feat)
        self.bn2 = nn.BatchNorm1d(self.num_elements * self.feat_dim * self.in_feat)
        self.bn3 = nn.BatchNorm1d(self.num_elements * self.feat_dim * self.in_feat)

        self.dropout = nn.Dropout(self.do_prob)

        self.activation=nn.Tanh()

        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
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
        wavelet1, wavelet2=self.compute_wavelets_torch(self.adj)
        input_reshaped=input.view(-1,self.feat_dim,self.in_feat)
        out1=torch.matmul(input_reshaped, self.weight1)
        out1=torch.matmul(self.adj.reshape(1, N, N), out1)
        if bn:
            out1 = out1.view((-1, self.num_elements* self.feat_dim* self.in_feat))
            out1=self.bn1(out1)

        out1=out1.view((-1,self.num_elements,self.feat_dim,self.in_feat))
        out2=torch.matmul(input_reshaped, self.weight2)
        out2 = torch.matmul(wavelet1, out2)
        if bn:
            out2 = out2.view((-1, self.num_elements* self.feat_dim* self.in_feat))
            out2=self.bn2(out2)

        out2=out2.view((-1,self.num_elements,self.feat_dim,self.in_feat))
        out3=torch.matmul(input_reshaped, self.weight3)
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
        self.weight_spectral = Parameter(torch.FloatTensor(self.in_feat, self.in_feat))
        self.weight_attention = Parameter(torch.FloatTensor(self.in_feat))
        self.channel_mlp=create_mlp(dim_input=self.in_feat*2*self.feat_dim,dim_output=self.in_feat, arch=[])
        self.channel_mlp = nn.Sequential(*self.channel_mlp)
        self.ReLU=nn.ReLU()
        self.Tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)
        self.num_scores = 9
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.weight_spectral.size(1))
        self.weight_spectral.data.uniform_(-stdv, stdv)
        self.weight_attention.data.uniform_(-stdv, stdv)

    def compute_H_sp(self, input):
        out = torch.matmul(input, self.weight_spectral)
        out=torch.mean(out,dim=1)
        return self.ReLU(out)

    def forward(self,input):
        H_sp=self.compute_H_sp(input)
        output=torch.cat([H_sp.unsqueeze(1).expand(-1,self.num_scores ,-1,-1),input],dim=3)
        output=self.channel_mlp(output.reshape((output.shape[0],self.num_scores ,-1)))
        output=self.Tanh(output)
        output = torch.matmul(output, self.weight_attention)
        output=self.softmax(output)
        return output
