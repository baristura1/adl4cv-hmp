import torch
import numpy as np
from torch import nn
from torch.nn import Module

class AttModule(Module):
    def __init__(self, in_n, out_n, kernel, feat_dim, hidden_size):
        """
        :param in_n: however many poses are considered
        :param out_n: number of future poses to be predicted
        :param kernel: key size, query size
        :param feat_dim:
        :param hidden_size:
        """
        super(AttModule, self).__init__()
        self.in_n = in_n
        self.out_n = out_n
        self.kernel = kernel
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.qNet = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_size, kernel_size=...),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=...),
                                  nn.ReLU())

        self.kNet = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_size, kernel_size=...),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=...),
                                  nn.ReLU())

        ##############
        # GCN MODULE #
        ##############

    def forward(self, x):
        """
        :param x: of size [batch_size, sequence_length, feature_dimension]
        :return:
        """
        x = x[:, :self.in_n, :]
        query_source = x[:, -self.out_n:, :]
        key_source = x[:, :(self.in_n - self.out_n), :]
        query_source = torch.from_numpy(query_source)
        key_source = torch.from_numpy(key_source)
        batch_size = x.shape[0]

        value_len = self.kernel + self.out_n
        value_seq = self.in_n - self.kernel - self.out_n + 1
        value_idx = np.expand_dims(np.arange(value_len), axis=0) + np.expand_dims(np.arange(value_seq), axis=1)
        value_mat = x[:, value_idx, :]
        value_mat = torch.from_numpy(value_mat)
        dct_mat, idct_mat = self.get_dct_mat(value_len)
        dct_mat = torch.from_numpy(dct_mat)
        value_dct = torch.matmul(value_mat.transpose(2,3), dct_mat)
        #att_final = torch.sum(value_dct, dim=1)

        query = self.qNet(query_source.transpose(1,2))
        key = self.kNet(key_source.transpose(1,2))

        att_scores_flat = torch.matmul(query.transpose(1,2), key)
        norm_factor = torch.sum(scores, dim=-1)
        att_scores = att_scores_flat.transpose(1,2).squeeze(2) / norm_factor
        att_final = torch.matmul(att_scores.reshape(batch_size,1,1,value_seq), value_dct.reshape(batch_size, self.feat_dim, value_seq, value_len))
        att_final = att_final.squeeze(2)
        padded_idx = np.arange(-self.kernel, 0, 1) + [-1] * self.out_n
        padded_query = x[:, padded_idx, :]
        padded_query = padded_query.transpose(1, 2)
        gcn_inp = torch.cat((padded_query, att_final),dim=2)



    def get_dct_mat(N):
        """
        Adopted from github.com/wei-mao-2019/HisRepItself
        :return:
        """
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        return dct_m, idct_m
