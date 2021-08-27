import torch
import torch.nn as nn
import torch.nn.functional as F

Norm_dict = {"layer" : nn.LayerNorm,
             "batch" : nn.BatchNorm1d}

Activation_dict = {"gelu" : F.gelu,
                   "relu" : F.relu}

class EncoderLayer(nn.Module):


    def __init__(self,
                 attention,
                 d_model,
                 dim_feedforward=None, 
                 feedforward_dropout=0.1,
                 activation = "gelu",
                 norm_type  = "layer",
                 forward_kernel_size=1):

        super(EncoderLayer, self).__init__()
		
        self.norm_type = norm_type
        # 输入是经过了norm的，所以不像其他代码，这里有prenorm
        # norm只发生在两个地方. 1. attention（x）+x  2. x+feedforward(x)  有prenorm的，都是在第二部之后，没有norm
        # ======================== 第一部分， self_attention ==============================
        self.self_attn = attention
        self.attn_norm = Norm_dict[norm_type](d_model)
		
        # ======================== 第二部分，  feedforward   ==============================

        self.dim_feedforward = dim_feedforward or 4*d_model
        self.forward_kernel_size = forward_kernel_size

        self.ffd_conv1 = nn.Conv1d(in_channels  = d_model, 
                                   out_channels = self.dim_feedforward, 
                                   kernel_size  = self.forward_kernel_size)
								   
        self.ffd_activation = Activation_dict[activation]
        self.ffd_dropout1 = nn.Dropout(feedforward_dropout)

        self.ffd_conv2 = nn.Conv1d(in_channels   = self.dim_feedforward,
                                   out_channels  = d_model, 
                                   kernel_size   = self.forward_kernel_size)

        self.ffd_dropout2 = nn.Dropout(feedforward_dropout)
        self.ffd_norm = Norm_dict[norm_type](d_model)





    def forward(self, x):
        # ======================== 第一部分， self_attention ==============================
        # 输入维度 B L C 
        new_x, attn = self.self_attn(x, x, x)
        x  =  x + new_x
        if self.norm_type == "layer":
            x = self.attn_norm(x).permute(0, 2, 1)
        else :
            x = self.attn_norm(x.permute(0, 2, 1))
        # 输入维度 B C L
        # ======================== 第二部分，  feedforward   ==============================
        forward_padding_size = int(self.forward_kernel_size/2)
        paddding_x   = nn.functional.pad(x, 
                                         pad=(forward_padding_size, forward_padding_size),
                                         mode='replicate')
        y            = self.ffd_dropout1(self.ffd_activation(self.ffd_conv1(paddding_x)))  

        paddding_y   = nn.functional.pad(y, 
                                         pad=(forward_padding_size, forward_padding_size),
                                         mode='replicate')    
        y            = self.ffd_dropout2(self.ffd_conv2(paddding_y))

        y = x + y #[B,C,L]
        if self.norm_type == "layer":

            y = self.ffd_norm(y.permute(0, 2, 1))
        else :
            y = self.ffd_norm(y).permute(0, 2, 1)		

        return y, attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, conv_norm = "batch", conv_activation = "relu"):
        super(ConvLayer, self).__init__()
        """
        专门用来降低长度的convblock，默认kernel=3，

        """
        self.norm_type = conv_norm

        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_out,
                                  kernel_size=3)

        self.normConv = Norm_dict[conv_norm](c_out)


        self.conv_activation = Activation_dict[conv_activation]

        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        paddding_x   = nn.functional.pad(x.permute(0, 2, 1), 
                                         pad=(1, 1),
                                         mode='replicate')
        x = self.downConv(paddding_x)

        if self.norm_type == "layer":
            x = self.normConv(x.permute(0, 2, 1)).permute(0, 2, 1)
        else :
            x = self.normConv(x)

        x = self.conv_activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layers, conv_layers=None):
        super(Encoder, self).__init__()

        #self.encoder_layers = nn.ModuleList(encoder_layers)
        #self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None

        model_list = []

        if conv_layers is not None:
            length_conv = len(conv_layers)
            for i in range(length_conv):
                model_list.append(encoder_layers[i])
                model_list.append(conv_layers[i])
            model_list.append(encoder_layers[-1])
            self.all_layers = nn.ModuleList(model_list)
        else:
            self.all_layers = nn.ModuleList(encoder_layers)



    def forward(self, x):
        # x [B, L, D]
        #attns = []
        #if self.conv_layers is not None:
        #    for encoder_layer, conv_layer in zip(self.encoder_layers, self.conv_layers):
        #        x, attn = encoder_layer(x)
        #        x = conv_layer(x)
        #        attns.append(attn)

        #    x, attn = self.encoder_layers[-1](x)
        #    attns.append(attn)
        #else:
        #    for encoder_layer in self.encoder_layers:
        #        x, attn = encoder_layer(x)
        #        attns.append(attn)
        attns = []


        for layer in self.all_layers:
            if isinstance(layer, EncoderLayer):
                x, attn = layer(x)
                attns.append(attn)
            else:
                x = layer(x)

        return x, attns