#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from utils import data_utils

parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1
parent = parent.tolist()
joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31]).tolist()
for idx in sorted(joint_to_ignore, reverse=True):
    del parent[idx]

parents = np.array(parent)
parents = np.argsort(parents)

def bonelength_error(batch_pred, batch_gt):

    bone_gt = batch_gt - batch_gt[:, :, parents, :] + 1e-8
    bonelengths_gt = torch.sqrt(torch.sum(torch.square(bone_gt), axis=-1))
    bone_pred = batch_pred - batch_pred[:, :, parents, :] + 1e-8
    bonelengths_pred = torch.sqrt(torch.sum(torch.square(bone_pred), axis=-1))

    bll=torch.mean(torch.norm(bonelengths_gt-bonelengths_pred,2,1))

    #oneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.square(bonelengths_pred - bonelengths_gt), axis=-1), axis=-1))

    return bll


def mpjpe_error(batch_pred,batch_gt): 




    
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))
    
    
def euler_error(ang_pred, ang_gt):

    # only for 32 joints
    
    dim_full_len=ang_gt.shape[2]

    # pred_expmap[:, 0:6] = 0
    # targ_expmap[:, 0:6] = 0
    pred_expmap = ang_pred.contiguous().view(-1,dim_full_len).view(-1, 3)
    targ_expmap = ang_gt.contiguous().view(-1,dim_full_len).view(-1, 3)

    pred_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(pred_expmap))
    pred_eul = pred_eul.view(-1, dim_full_len)

    targ_eul = data_utils.rotmat2euler_torch(data_utils.expmap2rotmat_torch(targ_expmap))
    targ_eul = targ_eul.view(-1, dim_full_len)
    mean_errors = torch.mean(torch.norm(pred_eul - targ_eul, 2, 1))

    return mean_errors




