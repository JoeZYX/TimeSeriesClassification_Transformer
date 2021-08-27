import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

# TODO 所有循环结构应该呈现灵活性，每一层都不能一样！
activation_dict = {"relu"   : nn.ReLU}

class TokenEmbedding(nn.Module):
    def __init__(self,
                 c_in, 
                 token_d_model,
                 kernel_size = 3, 
                 stride = 1, 
                 conv_bias=False,
                 activation="relu",
                 norm = "batch",
                 n_conv_layers=1,
                 in_planes=None,
                 max_pool=False,
                 pooling_kernel_size=3, 
                 pooling_stride=2,
                 pooling_padding=1):
        """
        c_in  : 模型输入的维度
        token_d_model ： embedding的维度  TODO看看后面是需要被相加还是被cat
        kernel_size   : 每一层conv的kernel大小
    
        """
        super(TokenEmbedding, self).__init__()
        in_planes = in_planes or int(token_d_model/2)
        n_filter_list = [c_in] + [in_planes for _ in range(n_conv_layers - 1)] + [token_d_model]
        padding = int(kernel_size/2)

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(in_channels  =  n_filter_list[i], 
                          out_channels =  n_filter_list[i + 1],
                          kernel_size  =  kernel_size,
                          padding      =  padding,
                          stride       =  stride,
                          bias         =  conv_bias,#),
                          padding_mode = 'replicate'),
                nn.Identity() if norm is None else nn.BatchNorm1d(n_filter_list[i + 1]),
                nn.Identity() if activation is None else activation_dict[activation](),
                nn.MaxPool1d(kernel_size = pooling_kernel_size,
                             stride      = pooling_stride,
                             padding     = pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)



    def forward(self, x):

        x = self.conv_layers(x.permute(0, 2, 1)).transpose(1,2)
        return x

    def sequence_length(self, length=100, n_channels=3):
        return self.forward(torch.zeros((1, length,n_channels))).shape[1]


class PositionalEmbedding(nn.Module):
    """
    input shape should be (batch, seq_length, feature_channel)
    
    """
    def __init__(self, pos_d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        
        pe = torch.zeros(max_len, pos_d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, pos_d_model, 2).float() * -(math.log(10000.0) / pos_d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)# [1, max_len, pos_d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # select the the length same as input


    def vis_pos_heat(self, length):
        heat = self.pe[:, :length]
        plt.figure(figsize=(15,5))
        sns.heatmap(heat.detach().numpy()[0], linewidth=0)
        plt.ylabel("length")
        plt.xlabel("embedding")
