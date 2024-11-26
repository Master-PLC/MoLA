import torch.nn as nn

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Decoder, Encoder
from models.Transformer import Model as Transformer
from tuners.adapter.Layers import (DecoderLayerWithAdapter,
                                   EncoderLayerWithAdapter)


class Model(Transformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_adapters = configs.label_pred_len
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
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.decoder = Decoder(
                [
                    DecoderLayerWithAdapter(
                        AttentionLayer(
                            FullAttention(
                                True, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        AttentionLayer(
                            FullAttention(
                                False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                            ),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        num_adapters=self.num_adapters,
                        r=configs.r,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
