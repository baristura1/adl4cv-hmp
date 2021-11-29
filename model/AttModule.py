import torch
import numpy as np
from torch import nn
from torch.nn import Module
from model import GCN
import utils.util as util


class AttModule(Module):
    def __init__(self, in_n, out_n, kernel, feat_dim, hidden_size):
        """
        :param in_n: However many poses are considered
        :param out_n: Number of future poses to be predicted
        :param kernel: Key size, query size
        :param feat_dim: Dimensionality of each pose
        :param hidden_size: Determines the hiddens of qNet, kNet and GraphConvs
        """
        super(AttModule, self).__init__()
        self.in_n = in_n
        self.out_n = out_n
        self.kernel = kernel
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.qNet = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_size, kernel_size=6),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=5),
                                  nn.ReLU())

        self.kNet = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_size, kernel_size=6),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=5),
                                  nn.ReLU())
        self.gcn = GCN.GCN(in_feat=(self.kernel+self.out_n)*2, hidden_feat=self.hidden_size, num_blocks=6,
                       dropout=0.3, feat_dim=self.feat_dim)

    def forward(self, x):
        """
        :param x: of shape [batch_size, sequence_length, feature_dimension]
        :return:
        """
        x = x[:, :self.in_n, :]
        query_source = x[:, -self.out_n:, :]
        key_source = x[:, :(self.in_n - self.out_n), :]
        query_source = query_source
        key_source = key_source
        batch_size = x.shape[0]

        value_len = self.kernel + self.out_n
        value_seq = self.in_n - self.kernel - self.out_n + 1
        value_idx = np.expand_dims(np.arange(value_len), axis=0) + np.expand_dims(np.arange(value_seq), axis=1)
        value_mat = x[:, value_idx, :]
        dct_mat, idct_mat = util.get_dct_matrix(self.kernel + self.out_n)
        dct_mat = torch.from_numpy(dct_mat).float()
        idct_mat = torch.from_numpy(idct_mat).float()
        value_dct = torch.matmul(value_mat.transpose(2, 3), dct_mat)

        query = self.qNet(query_source.transpose(1, 2))
        key = self.kNet(key_source.transpose(1, 2))

        att_scores_flat = torch.matmul(query.transpose(1, 2), key)
        norm_factor = torch.sum(att_scores_flat, dim=-1)
        att_scores = att_scores_flat.transpose(1, 2).squeeze(2) / norm_factor
        att_final = torch.matmul(att_scores.reshape(batch_size, 1, 1, value_seq),
                                 value_dct.reshape(batch_size, self.feat_dim, value_seq, value_len))
        att_final = att_final.squeeze(2)
        padded_idx = list(np.arange(-self.kernel, 0, 1)) + [-1] * self.out_n
        padded_query = x[:, padded_idx, :]
        padded_query = torch.matmul(padded_query.transpose(1,2), dct_mat) # there was a transpose(1,2)
        gcn_inp = torch.cat((padded_query, att_final), dim=2)
        gcn_output = self.gcn(gcn_inp)
        output = torch.matmul(gcn_output[:,:,:(self.kernel+self.out_n)], idct_mat)

        return output


