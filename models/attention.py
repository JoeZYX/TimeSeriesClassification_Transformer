
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


####################             Mask                   #########################################

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

Mask_dict = {"Triangular"     :TriangularCausalMask}





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
                 d_model, 
                 n_heads, 
                 d_keys=None, 
                 d_values=None, 
                 causal_kernel_size=3, 
                 value_kernel_size = 1,
                 projection_dropout=0.1):
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
        self.d_keys = d_keys or (d_model//n_heads)                                                   # 每个head中，key和query的维度
        self.d_values = d_values or (d_model//n_heads)                                               # 每个head中，value 的维度, 一般情况应该和key一样

        # 因为是时间序列，这里采取的causal attention，通过设置kernel的大小，可以是linear
        self.causal_kernel_size = causal_kernel_size                                                 # 提取key和query的kernel大小，当等于1时，就是linear，当大于1时就是conv
        self.value_kernel_size  = value_kernel_size                                                  # 提取value的kernel大小，同上
        self.projection_dropout = projection_dropout

        # 初始化4个projection，分别时key，query， value以及最后新value的out的projection
        self.query_projection = nn.Conv1d(in_channels = d_model,
                                          out_channels = self.d_keys*self.n_heads, 
                                          kernel_size  = self.causal_kernel_size)


        self.key_projection = nn.Conv1d(in_channels = d_model,
                                        out_channels = self.d_keys*self.n_heads, 
                                        kernel_size  = self.causal_kernel_size)


        self.value_projection = nn.Conv1d(in_channels= d_model,
                                          out_channels=self.d_values * self.n_heads, 
                                          kernel_size = self.value_kernel_size) 
										  
        self.inner_attention = attention

        self.out_projection = nn.Conv1d(in_channels=self.d_values * self.n_heads,                    # 与前三个projection的输入维度不一样，因为这里的输入时attention后的新value
                                        out_channels=d_model,                                        # 由于有skip的机制，所以整个attention的输入和输出要保持一直
                                        kernel_size = self.value_kernel_size)                       
        self.proj_drop = nn.Dropout(projection_dropout)


    def forward(self, queries, keys, values):

        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape                                                                   # 理论上所有的L_和I_是一模一样的
        H = self.n_heads

        # # 以上 B L C 中的C是包含了所有Head的特征，映射之后拆分为，每个head的特征，也就是， [B, L, H, C] 
        #  ========================== value projection ==========================
        value_padding_size   = int(self.value_kernel_size/2)
        paddding_values      = nn.functional.pad(values.permute(0, 2, 1), 
                                                 pad=(value_padding_size, value_padding_size),
                                                 mode='replicate')
        values               = self.value_projection(paddding_values).permute(0, 2, 1)  # B L C
        values               = values.view(B, L_V, H, -1)

        # ========================== query  keys projection ==========================
        queries_padding_size = int(self.causal_kernel_size/2)

        paddding_queries     = nn.functional.pad(queries.permute(0, 2, 1), 
                                                 pad=(queries_padding_size, queries_padding_size),
                                                 mode='replicate')
        queries              = self.query_projection(paddding_queries).permute(0, 2, 1) # B L C
        queries              = queries.view(B, L_Q, H, -1)

     
        paddding_keys        = nn.functional.pad(keys.permute(0, 2, 1), 
                                                 pad=(queries_padding_size, queries_padding_size),
                                                 mode='replicate')
        keys                 = self.key_projection(paddding_keys).permute(0, 2, 1) # B L C  
        keys                 = keys.view(B, L_K, H, -1)   


        # ========================== attention ==========================
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        out = out.view(B, L_V, -1)                                                                 # TODO L_V?                                                 

        # ========================== Out Projection ==========================
        paddding_out        = nn.functional.pad(out.permute(0, 2, 1), 
                                                pad=(value_padding_size, value_padding_size),
                                                mode='replicate')
        out                 = self.out_projection(paddding_out).permute(0, 2, 1)

			
        out                 = self.proj_drop(out)
        return out, attn