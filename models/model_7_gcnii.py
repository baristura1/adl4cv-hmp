#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import math

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
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
                 joints_dim, in_channels, out_channels, residual=False
    ):
        super(ConvTemporalGraphical,self).__init__()

        self.time_dim = time_dim
        self.joints_dim = joints_dim
        self.in_channels = in_channels

        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim*time_dim, joints_dim*time_dim))
        stdv = 1. / math.sqrt(time_dim)
        self.Z.data.uniform_(-stdv,stdv)

        # todo: consider output channels during parameter init
        self.W = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        stdv = 1. / math.sqrt(in_channels)
        self.W.data.uniform_(-stdv, stdv)

        self.residual=residual

    def forward(self, x, h0 , lamda, alpha, l):

        N = x.shape[0]
        C = self.in_channels
        T = self.time_dim
        V = self.joints_dim

        x = x.view(-1, C, T * V)

        x = x.permute(0, 2, 1)

        h0=h0.view(-1, C, T * V)

        h0 = h0.permute(0, 2, 1)

        theta = math.log(lamda/l+1)

        hi = self.Z @ x

        support = (1-alpha)*hi+alpha*h0

        r = support

        output=theta*(support @ self.W)

        output = output+(1-theta)*r

        output=output.permute(0,2,1)

        output = output.view(-1, C, T, V)


        if self.residual:
            output = output+x

        return output.contiguous()

class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
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
                 in_channels_0,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2,(self.kernel_size[1] - 1) // 2)
        
        
        self.gcn=ConvTemporalGraphical(time_dim,joints_dim, in_channels, out_channels) # the convolution layer
        
        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()

        self.residual2 = nn.Sequential(nn.Conv2d(
            in_channels_0,
            in_channels,
            kernel_size=1,
            stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
        )
        
        self.prelu = nn.PReLU()

    def forward(self, x, x0 , lamda, alpha, l):
     #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x0=self.residual2(x0)
        x=self.gcn(x, x0, lamda, alpha, l)
        x=self.tcn(x)
        x=x+res
        x=self.prelu(x)
        return x

class CNN_layer(nn.Module): # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
                     ,nn.BatchNorm2d(out_channels),nn.Dropout(dropout, inplace=True)] 

        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
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
        
        super(Model,self).__init__()
        self.input_time_frame=input_time_frame
        self.output_time_frame=output_time_frame
        self.joints_to_consider=joints_to_consider
        self.st_gcnns=nn.ModuleList()
        self.n_txcnn_layers=n_txcnn_layers
        self.txcnns=nn.ModuleList()

        self.st_gcnns.append(ST_GCNN_layer(3,3,64,[1,1],1,input_time_frame,
                                           joints_to_consider,st_gcnn_dropout))
        self.st_gcnns.append(ST_GCNN_layer(64,64,32,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
            
        self.st_gcnns.append(ST_GCNN_layer(32,64,64,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))
                                               
        self.st_gcnns.append(ST_GCNN_layer(64,64,3,[1,1],1,input_time_frame,
                                               joints_to_consider,st_gcnn_dropout))                                               
                
                
                # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txcnns.append(CNN_layer(input_time_frame,output_time_frame,txc_kernel_size,txc_dropout)) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame,output_time_frame,txc_kernel_size,txc_dropout))

        self.prelu=nn.PReLU()
            
        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):

        lamda, alpha = 0.5, 0.1
        x_0=x
        x = self.st_gcnns[0](x, x_0, lamda, alpha, 1)
        x_0=x

        for i,gcn in enumerate(self.st_gcnns):
            if i==0:
                continue
            x = gcn(x, x_0, lamda, alpha, i + 1)
            
        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        
        x=self.prelus[0](self.txcnns[0](x))
        
        for i in range(1,self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) +x # residual connection
            
        return x
        





