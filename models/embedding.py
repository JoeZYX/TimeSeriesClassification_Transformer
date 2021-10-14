import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np

# TODO 所有循环结构应该呈现灵活性，每一层都不能一样！
activation_dict = {"relu"         : nn.ReLU,
                   "leakyrelu"    : nn.LeakyReLU,
                   "prelu"        : nn.PReLU,
                   "rrelu"        : nn.RReLU,
                   "elu"          : nn.ELU,
                   "gelu"         : nn.GELU,
                   "hardswish"    : nn.Hardswish,
                   "mish"         : nn.Mish}
Norm_dict = {"layer" : nn.LayerNorm,
             "batch" : nn.BatchNorm1d}


class DW_PW_projection(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, bias = False, padding_mode = "replicate"):
        super(DW_PW_projection, self).__init__()

        self.dw_conv1d = nn.Conv1d(in_channels  = c_in,
                                   out_channels = c_in,
                                   kernel_size  = kernel_size,
                                   padding      = int(kernel_size/2),
                                   groups       = c_in,
                                   stride       = stride,
                                   bias         = bias,  
                                   padding_mode = padding_mode)

        self.pw_conv1d = nn.Conv1d(in_channels  = c_in,
                                   out_channels = c_out,
                                   kernel_size  = 1,
                                   padding      = 0,
                                   groups       = 1,
                                   bias         = bias,  
                                   padding_mode = padding_mode)
    def forward(self, x):


        x  = self.dw_conv1d(x)
        x  = self.pw_conv1d(x)

        return x

class Forward_block(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride                 =  1, 
                 conv_bias              = False,
                 activation             = "relu",
                 norm_type              = "batch",
                 max_pool               = False,
                 pooling_kernel_size    = 3, 
                 pooling_stride         = 2,
                 pooling_padding        = 1,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        embedding的block 由 conv --> norm --> activation --> maxpooling组成
        """
        super(Forward_block, self).__init__() 
        if light_weight:
            self.conv = DW_PW_projection(c_in         = c_in, 
                                         c_out        = c_out,
                                         kernel_size  = kernel_size,
                                         stride       =  stride,
                                         bias         = conv_bias, 
                                         padding_mode = padding_mode)
        else:
            self.conv = nn.Conv1d(in_channels  =  c_in, 
                                  out_channels =  c_out,
                                  kernel_size  =  kernel_size,
                                  padding      =  int(kernel_size/2),
                                  stride       =  stride,
                                  bias         =  conv_bias,
                                  padding_mode =  padding_mode)
        self.norm_type   = norm_type
        self.norm        = Norm_dict[norm_type](c_out)
        self.activation  = activation_dict[activation]()
        self.max_pool    = max_pool
        if max_pool:
           self.maxpooling =  nn.MaxPool1d(kernel_size = pooling_kernel_size,
                                           stride      = pooling_stride,
                                           padding     = pooling_padding)
    def forward(self, x):

        x  = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.norm_type == "layer":
            x = self.activation(self.norm(x))
        else :
            x = self.activation(self.norm(x.permute(0, 2, 1)).permute(0, 2, 1))

        if self.max_pool:
            x = self.maxpooling(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Freq_Forward_block(nn.Module):
    def __init__(self, 
                 c_in, 
                 c_out,  # 主要是把channel的dim压平
                 kernel_size, 
                 stride=1, 
                 bias = False, 
                 padding_mode = "replicate"):
        
        super(Freq_Forward_block, self).__init__()
        
        # depthwise
        self.dw_conv = nn.Conv2d(in_channels  = c_in,
                                 out_channels = c_in,
                                 kernel_size  = [kernel_size,kernel_size],
                                 padding      = [int(kernel_size/2),int(kernel_size/2)],
                                 groups       = c_in,
                                 stride       = [1,stride],  #缩短长度
                                 bias         = bias,  
                                 padding_mode = padding_mode)
        self.batch_norm_1 = nn.BatchNorm2d(c_in)
        self.act_1  = nn.ReLU()
        # pointwise
        self.pw_conv = nn.Conv2d(in_channels  = c_in,
                                 out_channels = c_out,    # 压平
                                 kernel_size  = 1,
                                 padding      = 0,
                                 stride       = 1,
                                 bias         = bias,  
                                 padding_mode = padding_mode)
        self.batch_norm_2 = nn.BatchNorm2d(c_out)
        self.act_2  = nn.ReLU()
        
    def forward(self, x):

        x  = self.dw_conv(x)
        x  = self.batch_norm_1(x)
        x  = self.act_1(x)

        x  = self.pw_conv(x)
        x  = self.batch_norm_2(x)
        x  = self.act_2(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self,
                 c_in, 
                 token_d_model,
                 kernel_size            = 3, 
                 stride                 = 1, 
                 conv_bias              = False,
                 activation             = "relu",
                 norm_type              = "batch",
                 n_conv_layers          = 1,
                 in_planes              = None,
                 max_pool               = False,
                 pooling_kernel_size    = 3, 
                 pooling_stride         = 2,
                 pooling_padding        = 1,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        c_in  : 模型输入的维度
        token_d_model ： embedding的维度  TODO看看后面是需要被相加还是被cat
        kernel_size   : 每一层conv的kernel大小
    
        """
        super(TokenEmbedding, self).__init__()
        in_planes = in_planes or int(token_d_model/2)
        n_filter_list = [c_in] + [in_planes for _ in range(n_conv_layers - 1)] + [token_d_model]
        padding = int(kernel_size/2)


        self.conv_layers = []
        for i in range(n_conv_layers):
            self.conv_layers.append(Forward_block(c_in                = n_filter_list[i],
                                                  c_out               = n_filter_list[i + 1], 
                                                  kernel_size         = kernel_size,
                                                  stride              = stride, 
                                                  conv_bias           = conv_bias,
                                                  activation          = activation,
                                                  norm_type           = norm_type,
                                                  max_pool            = max_pool,
                                                  pooling_kernel_size = pooling_kernel_size, 
                                                  pooling_stride      = pooling_stride,
                                                  pooling_padding     = pooling_padding,
                                                  padding_mode        = padding_mode,
                                                  light_weight        = light_weight))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv1d):
        #        nn.init.kaiming_normal_(m.weight)



    def forward(self, x):


        for layer in self.conv_layers:
            x = layer(x)
        return x

    def sequence_length(self, length=100, n_channels=3):
        return self.forward(torch.zeros((1, length,n_channels))).shape[1]



class Freq_TokenEmbedding(nn.Module):
    def __init__(self,
                 c_in, 
                 token_d_model,
                 kernel_size            = 3, 
                 stride                 = 1,  #横向方向缩短距离
                 conv_bias              = False,
                 n_conv_layers          = 1,
                 f_max                  = 100,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        c_in  : 模型输入的维度
        token_d_model ： embedding的维度  TODO看看后面是需要被相加还是被cat
        kernel_size   : 每一层conv的kernel大小
    
        """
        super(Freq_TokenEmbedding, self).__init__()

        n_filter_list = [c_in] + [max(1,int(100/2**(i+1))) for i in range(n_conv_layers - 1)] + [1]
        print(n_filter_list)
        self.conv_layers = []
        for i in range(n_conv_layers):
            self.conv_layers.append(Freq_Forward_block(c_in           = n_filter_list[i], 
                                                       c_out          = n_filter_list[i + 1],  # 主要是把channel的dim压平
                                                       kernel_size    = kernel_size, 
                                                       stride         = stride, 
                                                       bias           = conv_bias,
                                                       padding_mode   = padding_mode))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.conv = nn.Conv1d(in_channels  =  self.channel(c_in = c_in, freq = int(f_max/2), length=100), 
                              out_channels =  token_d_model,
                              kernel_size  =  kernel_size,
                              padding      =  int(kernel_size/2),
                              stride       =  1,
                              bias         =  conv_bias,
                              padding_mode =  padding_mode)
        self.norm        = nn.LayerNorm(token_d_model)
        self.activation  = nn.ReLU()
    def forward(self, x):


        for layer in self.conv_layers:
            x = layer(x)

        x = torch.squeeze(x, 1)

        x = self.conv(x) # B C L
        x = self.activation(self.norm(x.permute(0, 2, 1)))

        return x
    
    def sequence_length(self, c_in = 100, freq = 50, length=100):
        x =  torch.rand(1,c_in,freq,length).float()
        for layer in self.conv_layers:
            x = layer(x)
        return x.shape[3]

    def channel(self, c_in = 100, freq = 50, length=100):
        x =  torch.rand(1,c_in,freq,length).float()
        for layer in self.conv_layers:
            x = layer(x)
        print("channel ," x.shape[2])
        return x.shape[2]

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

