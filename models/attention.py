
# transformer的一层分为两个部分， 第一部分是selfattention，第二个部分是feedforward
# 以下代码首先编写self attention
# self attention 

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import random
from math import sqrt

Norm_dict = {"layer" : nn.LayerNorm,
             "batch" : nn.BatchNorm1d}

class DW_PW_projection(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, bias = False, padding_mode = "replicate"):
        super(DW_PW_projection, self).__init__()

        self.dw_conv1d = nn.Conv1d(in_channels  = c_in,
                                   out_channels = c_in,
                                   kernel_size  = kernel_size,
                                   padding      = int(kernel_size/2),
                                   groups       = c_in,
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

####################             Mask                   #########################################

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.ones((L, L)).to(device)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
            
    @property   
    def mask(self):
        return self._mask

class LocalSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            window_size = math.ceil(1.2*np.log2(L)/2)  #halb
            mask = torch.ones((L, L)).to(device)
            mask = torch.triu(mask,-window_size).T
            mask = torch.triu(mask,-window_size)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)  
    @property            
    def mask(self):
        return self._mask

class LocalLogSymmetryMask():
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            mask = torch.zeros((L, L), dtype=torch.float).to(device)
            for i in range(L):
                mask[i] = self.row_mask(i, L)
            mask = mask==0
            mask = torch.unsqueeze(mask, 0)
            self._mask = mask.expand(B, 1, L, L).to(device)

            
    def row_mask(self,index, L):
        local_window_size = math.ceil(np.log2(L)/2) # 1/2 window size
        # 对当前行的index 行 进行初始化
        mask = torch.zeros((L), dtype=torch.float)

        if((index - local_window_size + 1) < 0):
            mask[:index] = 1 # Local attention
        else:
            mask[index - local_window_size + 1:(index + 1)] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index - local_window_size + 1 - int(1.5**i)
                if new_index >= 0:
                    mask[new_index] = 1
                else:
                    break
                    
        if ((index + local_window_size-1 )>=L):
            mask[index:] = 1 
        else:
            mask[index:index+local_window_size] = 1  # Local attention

            for i in range(0, math.ceil(10*np.log2(L))):
                new_index = index + local_window_size-1 +int(1.5**i)
                if new_index < L:
                    mask[new_index] = 1
                else:
                    break
        return mask               

    @property          
    def mask(self):
        return self._mask

Mask_dict = {"Triangular"     :TriangularCausalMask,
             "LocalSymmetry"  :LocalSymmetryMask,
             "Full"           :FullMask,
             "LocLogSymmetry" :LocalLogSymmetryMask}





####################         Mask Attention      ###############################
class MaskAttention(nn.Module):
    def __init__(self, 
                 mask_flag=True, 
                 mask_typ = "Triangular",
                 attention_dropout=0.1, 
                 output_attention=False):
        """
        mask_flag ： 是否使用mask，如果不使用，那么就是全局mask
        mask_typ  ： 如果使用mask，哪种？
        attention_dropout ： attention之后 score的dropout
        output_attention  ： bool，是否输出attentionmap
        """
        super(MaskAttention, self).__init__()
        self.mask_typ         = mask_typ
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.attn_drop        = nn.Dropout(attention_dropout)
				
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]

        返回的是两个东西
        1.  attn_values : 新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
 


        queries = queries.permute(0, 2, 1, 3)                                                     # [batch, heads, length, chanell]
        keys    = keys.permute(0, 2, 3, 1)                                                        # [batch, heads, chanell, length]
        attn    = torch.matmul(queries, keys)                                                     
        scale   =  1./math.sqrt(E) 
        attn    = scale * attn
        
        if self.mask_flag:
            attn_mask = Mask_dict[self.mask_typ](B, L, device=queries.device)
            attn.masked_fill_(attn_mask.mask, -np.inf)                                       #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf        


        attn = self.attn_drop(torch.softmax(attn , dim=-1))
        
        values = values.permute(0, 2, 1, 3)                                                       # [batch, heads, length, chanell]
        attn_values = torch.matmul(attn, values).permute(0,2,1,3)                                 # [batch, length, heads, chanell]

        
        if self.output_attention:
            return (attn_values.contiguous(), attn)
        else:
            return (attn_values.contiguous(), None)


####################         Attention Layer      ###############################
class AttentionLayer(nn.Module):
    def __init__(self, 
                 attention, 
                 input_dim,
                 d_model, 
                 n_heads, 
                 d_keys               =  None, 
                 d_values             =  None, 
                 causal_kernel_size   =  3, 
                 value_kernel_size    =  1,
                 bias                 = False,
                 padding_mode         = 'replicate',
                 projection_dropout   =  0.1,
                 light_weight         = False):
        """

        attention          :    要进行什么样子的attention？Probmask？seasonal？还是全局的？ 默认就是full吧
        d_model            :    输入的维度
        n_heads            :    注意力的个数
        d_keys             ：    query和key的映射维度 ，默认是和d_model一样大
        d_values           ：    value的映射维度，默认是和d_model一样大
        causal_kernel_size :    是否通过local conv进行提取特征。 如果等于1， 就是linear. 如果大于1，就是1d conv
        value_kernel_size  :    和上面参数一致
        attention_dropout  ：    
        
	    """

        super(AttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.d_keys = d_keys or d_model                                                              # 每个head中，key和query的维度
        self.d_values = d_values or d_model                                                          # 每个head中，value 的维度, 一般情况应该和key一样

        # 因为是时间序列，这里采取的causal attention，通过设置kernel的大小，可以是linear
        self.causal_kernel_size = causal_kernel_size                                                 # 提取key和query的kernel大小，当等于1时，就是linear，当大于1时就是conv
        self.value_kernel_size  = value_kernel_size                                                  # 提取value的kernel大小，同上
        self.projection_dropout = projection_dropout

        # 初始化4个projection，分别时key，query， value以及最后新value的out的projection
        if light_weight:
            self.query_projection = DW_PW_projection(c_in         = input_dim, 
                                                     c_out        = self.d_keys, 
                                                     kernel_size  = self.causal_kernel_size,
                                                     bias         = bias, 
                                                     padding_mode = padding_mode)
        else:
            self.query_projection = nn.Conv1d(in_channels  = input_dim,
                                              out_channels = self.d_keys, 
                                              kernel_size  = self.causal_kernel_size,
                                              padding      =  int(self.causal_kernel_size/2),
                                              bias         =  bias,  
                                              padding_mode = padding_mode)

        if light_weight:
            self.key_projection = DW_PW_projection(c_in         = input_dim, 
                                                   c_out        = self.d_keys, 
                                                   kernel_size  = self.causal_kernel_size,
                                                   bias         = bias, 
                                                   padding_mode = padding_mode)
        else:
            self.key_projection = nn.Conv1d(in_channels  = input_dim,
                                            out_channels = self.d_keys, 
                                            kernel_size  = self.causal_kernel_size,
                                            padding      =  int(self.causal_kernel_size/2),
                                            bias         =  bias,  
                                            padding_mode = padding_mode)

        if light_weight:
            self.value_projection = DW_PW_projection(c_in         = input_dim, 
                                                     c_out        = self.d_values, 
                                                     kernel_size  = self.value_kernel_size,
                                                     bias         = bias, 
                                                     padding_mode = padding_mode)
        else:
            self.value_projection = nn.Conv1d(in_channels  = input_dim,
                                              out_channels = self.d_values , 
                                              kernel_size  = self.value_kernel_size,
                                              padding      =  int(self.value_kernel_size/2),
                                              bias         =  bias,  
                                              padding_mode = padding_mode)
										  
        self.inner_attention = attention


        if light_weight:
            self.out_projection = DW_PW_projection(c_in         = self.d_values, 
                                                   c_out        = d_model, 
                                                   kernel_size  = self.value_kernel_size,
                                                   bias         = bias, 
                                                   padding_mode = padding_mode)
        else:
            self.out_projection = nn.Conv1d(in_channels  = self.d_values ,                                  # 与前三个projection的输入维度不一样，因为这里的输入时attention后的新value
                                            out_channels = d_model,                                        # 由于有skip的机制，所以整个attention的输入和输出要保持一直
                                            kernel_size  = self.value_kernel_size,
                                            padding      = int(self.value_kernel_size/2),
                                            bias         = bias,  
                                            padding_mode = padding_mode)
        self.proj_drop = nn.Dropout(projection_dropout)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv1d):
        #        nn.init.kaiming_normal_(m.weight)


    def forward(self, queries, keys, values):

        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape                                                                   # 理论上所有的L_和I_是一模一样的
        H = self.n_heads

        # # 以上 B L C 中的C是包含了所有Head的特征，映射之后拆分为，每个head的特征，也就是， [B, L, H, C] 
        #  ========================== value projection ==========================
        values               = self.value_projection(values.permute(0, 2, 1)).permute(0, 2, 1)
        values               = values.view(B, L_V, H, -1)

        # ========================== query  keys projection ==========================
        queries              = self.query_projection(queries.permute(0, 2, 1)).permute(0, 2, 1)
        queries              = queries.view(B, L_Q, H, -1)

        keys                 = self.key_projection(keys.permute(0, 2, 1)).permute(0, 2, 1)
        keys                 = keys.view(B, L_K, H, -1)   


        # ========================== attention ==========================
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L_V, -1)                                                                 # TODO L_V?                                                 

        # ========================== Out Projection ==========================

        out                 = self.out_projection(out.permute(0, 2, 1)).permute(0, 2, 1)
			
        out                 = self.proj_drop(out)
        return out, attn