import torch.nn as nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder
from models.iTransformer import Model as iTransformer
from tuners.adapter.Layers import EncoderLayerWithAdapter


class Model(iTransformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_adapters = configs.label_pred_len // configs.pred_len
        self.encoder = Encoder(
            [
                EncoderLayerWithAdapter(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    num_adapters=self.num_adapters,
                    r=configs.r,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
