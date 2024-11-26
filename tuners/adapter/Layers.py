import torch.nn as nn

from layers.Transformer_EncDec import DecoderLayer, EncoderLayer, FFNLayer


class EncoderLayerWithAdapter(EncoderLayer):
    def __init__(self, attention, d_model, d_ff=None, num_adapters=1, r=None, dropout=0.1, activation="relu"):
        super(EncoderLayerWithAdapter, self).__init__(attention, d_model, d_ff, dropout, activation)
        self.num_adapters = num_adapters
        r = r or d_model // 4
        self.adapter = nn.ModuleList(
            [FFNLayer(d_model, r, dropout, activation) for _ in range(num_adapters)]
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.ffn(y)

        tuner_index = kwargs.get('tuner_index', 0)
        init = kwargs.get('init', False)
        if not init:
            y = y + self.adapter[tuner_index](y)

        return self.norm2(x + y), attn


class DecoderLayerWithAdapter(DecoderLayer):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, num_adapters=1, r=None, dropout=0.1, activation="relu"):
        super(DecoderLayerWithAdapter, self).__init__(self_attention, cross_attention, d_model, d_ff, dropout, activation)
        self.num_adapters = num_adapters
        r = r or d_model // 4
        self.adapter = nn.ModuleList(
            [FFNLayer(d_model, r, dropout, activation) for _ in range(num_adapters)]
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, **kwargs):
        
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.ffn(y)

        tuner_index = kwargs.get('tuner_index', 0)
        init = kwargs.get('init', False)
        if not init:
            y = y + self.adapter[tuner_index](y)

        return self.norm3(x + y)