import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.embedding import TokenEmbedding
from models.encoder import EncoderLayer, ConvLayer, Encoder
from models.attention import AttentionLayer, MaskAttention


class TSCtransformer(nn.Module):
    def __init__(self, args):
        super(TSCtransformer, self).__init__()
        # all parameters are saved in self.args
        self.args = args

        # ================================ Embedding part ================================
        if self.args.token_d_model is not None:
            self.value_embedding = TokenEmbedding(c_in                 = args.c_in, 
                                                  token_d_model        = args.token_d_model,
                                                  kernel_size          = args.token_kernel_size, 
                                                  stride               = args.token_stride, 
                                                  conv_bias            = args.token_conv_bias,
                                                  activation           = args.token_activation,
                                                  norm                 = args.token_norm,
                                                  n_conv_layers        = args.token_n_layers,
                                                  in_planes            = args.token_in_planes,
                                                  max_pool             = args.token_max_pool,
                                                  pooling_kernel_size  = args.token_pool_kernel_size, 
                                                  pooling_stride       = args.token_pool_stride,
                                                  pooling_padding      = args.token_pool_pad)


            sequence_length = self.value_embedding.sequence_length(length       =  args.input_length, 
                                                                   n_channels   =  args.c_in)



        if args.positional_embedding != 'none':
            if args.positional_embedding == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, args.token_d_model), requires_grad=True)
                nn.init.trunc_normal_(self.pos_embedding, std=0.2)
            else:
                self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(sequence_length, args.token_d_model), requires_grad=False)
        else:
            self.pos_embedding = None

        self.input_embedding_dropout = nn.Dropout(p = args.input_embedding_dropout)
        # ================================ Encoding part ================================

        self.encoder = Encoder([EncoderLayer(AttentionLayer(attention          = MaskAttention(mask_flag          = args.mask_flag, 
                                                                                               mask_typ           = args.mask_typ,
                                                                                               attention_dropout  = args.attention_dropout, 
                                                                                               output_attention   = args.output_attention ),

                                                            d_model            = args.token_d_model, 
                                                            n_heads            = args.n_heads,
                                                            d_keys             = args.d_keys, 
                                                            d_values           = args.d_values, 
                                                            causal_kernel_size = args.causal_kernel_size, 
                                                            value_kernel_size  = args.value_kernel_size,
                                                            projection_dropout = args.projection_dropout),

                                             d_model             = args.token_d_model,
                                             dim_feedforward     = args.feedforward_dim, 
                                             feedforward_dropout = args.feedforward_dropout,
                                             activation          = args.feedforward_activation,
                                             norm_type           = args.feedforward_norm_type,
                                             forward_kernel_size = args.forward_kernel_size) for l in range(args.e_layers)],

                               [ConvLayer( c_in            = args.token_d_model, 
                                           c_out           = args.token_d_model,  
                                           conv_norm       = args.conv_norm, 
                                           conv_activation = args.conv_activation ) for l in range(args.e_layers-1)] if args.distil else None
                               )

        # ================================ Prediction part ================================
        # Variante 1 --------------
        #self.attention_pool = nn.Linear(args.token_d_model, 1)
        #self.classes_prediction = nn.Linear(args.token_d_model, args.num_classes)


        # Variante 2 --------------
        self.donwconv = nn.Conv1d( in_channels    = args.token_d_model,  
                                   out_channels   = 1, 
                                   kernel_size    = 3, 
                                   stride         = 1,
                                   padding        = 1,
                                   bias=True)
        if args.distil:
            final_length = int(args.input_length/(2*(args.e_layers-1)))
        else:
            final_length = args.input_length
        self.classes_prediction = nn.Linear(in_features=final_length, out_features=args.num_classes)



    def forward(self, x): 
        x = self.value_embedding(x)
        if self.pos_embedding is not None:
            x += self.pos_embedding
        x = self.input_embedding_dropout(x)

        x, attns = self.encoder(x)
        # Variante 1 --------------
        #x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        #x = self.classes_prediction(x)



        # Variante 2 --------------
        x = self.donwconv(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze()
        x = self.classes_prediction(x)



        return x, attns


    @staticmethod
    def sinusoidal_embedding(length, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(length)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
        