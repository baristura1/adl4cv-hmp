import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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


class GCBlock(nn.Module):

    def __init__(self, in_feat, dropout, feat_dim):
        """
        One block of GCN comprising of two graph convolutions and one skip connection.
        :param in_feat: second dim of input matrix H_i elem_of R^(KxF)
        :param dropout: probability
        :param feat_dim: first dim of input and output matrices, K
        """
        super(GCBlock, self).__init__()
        self.in_feat = in_feat
        self.do_prob = dropout
        self.feat_dim = feat_dim
        # self.out_dim is the same as self.in_dim, as dims are kept
        self.gc1 = GraphConvolution(self.in_feat, self.in_feat, self.feat_dim)
        self.bn1 = nn.BatchNorm1d(self.feat_dim*self.in_feat)

        self.gc2 = GraphConvolution(self.in_feat, self.in_feat, self.feat_dim)
        self.bn2 = nn.BatchNorm1d(self.feat_dim*self.in_feat)
        self.dropout = nn.Dropout(self.do_prob)
        self.activation = nn.Tanh()

    def forward(self, inp):
        out = self.gc1(inp)
        b, n, f = out.shape
        out = self.bn1(out.view(b, -1)).view(b, n, f)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.gc2(out)
        b, n, f = out.shape
        out = self.bn2(out.view(b, -1)).view(b, n, f)
        out = self.activation(out)
        out = self.dropout(out)

        return out + inp


class GCN(nn.Module):

    def __init__(self, in_feat, hidden_feat, num_blocks, dropout, feat_dim):
        """
        Complete graph convolutional network.
        :param in_feat: Input features, 2nd dim of the matrix obtained by concat'ing attention output and query DCT
        :param hidden_feat: Num of hidden features, design choice
        :param num_blocks: Determines how many GCBlocks are used
        :param dropout: Dropout probability.
        :param feat_dim: Feature dimensionality
        """
        super(GCN, self).__init__()
        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.num_blocks = num_blocks
        self.do_prob = dropout
        self.feat_dim = feat_dim

        self.gc1 = GraphConvolution(self.in_feat, self.hidden_feat, self.feat_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_feat*self.feat_dim)

        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append(GCBlock(self.hidden_feat, self.dropout, self.feat_dim))

        self.blocks = nn.ModuleList(self.blocks)

        self.gc_final = GraphConvolution(self.hidden_feat, self.in_feat, self.feat_dim)

        self.dropout = nn.Dropout(self.do_prob)
        self.activation = nn.Tanh()

    def forward(self, inp):
        out = self.gc1(inp)
        b, n, f = out.shape
        out = self.bn1(out.view(b, -1)).view(b, n, f)
        out = self.activation(out)
        out = self.dropout(out)

        for i in range(self.num_blocks):
            out = self.blocks[i](out)

        out = self.gc_final(out)

        return out + inp
