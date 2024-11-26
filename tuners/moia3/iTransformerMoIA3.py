import torch.nn as nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder
from models.iTransformer import Model as iTransformer
from tuners.moia3.Layers import AttentionLayerIA3, EncoderLayerWithIA3


class Model(iTransformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_group = configs.label_pred_len // configs.pred_len
        self.num_ia3 = configs.n_exp
        self.apply_attention = configs.apply_attention if hasattr(configs, "apply_attention") else False

        if not self.apply_attention:
            self.encoder = Encoder(
                [
                    EncoderLayerWithIA3(
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor, attention_dropout=configs.dropout, 
                                output_attention=configs.output_attention
                            ), 
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        num_group=self.num_group,
                        num_ia3=self.num_ia3,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model)
            )

        else:
            self.encoder = Encoder(
                [
                    EncoderLayerWithIA3(
                        AttentionLayerIA3(
                            FullAttention(
                                False, configs.factor, attention_dropout=configs.dropout, 
                                output_attention=configs.output_attention
                            ), 
                            configs.d_model, configs.n_heads, num_group=self.num_group, num_ia3=self.num_ia3
                        ),
                        configs.d_model,
                        configs.d_ff,
                        num_group=self.num_group,
                        num_ia3=self.num_ia3,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model)
            )
