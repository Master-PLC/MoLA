import torch
import torch.nn as nn

from layers.SelfAttention_Family import AttentionLayer
from layers.Transformer_EncDec import EncoderLayer, FFNLayer


class AttentionLayerIA3(AttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, num_ia3=1):
        super().__init__(attention, d_model, n_heads, d_keys, d_values)

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.ia3_lk = nn.ParameterList([nn.Parameter(torch.randn(d_keys)) for _ in range(num_ia3)])
        self.ia3_lv = nn.ParameterList([nn.Parameter(torch.randn(d_values)) for _ in range(num_ia3)])
        self.reset_parameters()

    def reset_parameters(self):
        for ia3_lk, ia3_lv in zip(self.ia3_lk, self.ia3_lv):
            nn.init.constant_(ia3_lk, 1.0)
            nn.init.constant_(ia3_lv, 1.0)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, **kwargs):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        tuner_index = kwargs.get("tuner_index", 0)
        init = kwargs.get("init", False)
        if not init:
            lk = self.ia3_lk[tuner_index]
            lv = self.ia3_lv[tuner_index]
            queries = lk * queries
            values = lv * values

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FFNLayerIA3(FFNLayer):
    def __init__(self, d_model, d_ff=None, num_ia3=1, dropout=0.1, activation="relu") -> None:
        super().__init__(d_model, d_ff, dropout, activation)
        d_ff = d_ff or 4 * d_model
        self.ia3_lff = nn.ParameterList([nn.Parameter(torch.randn(d_ff)) for _ in range(num_ia3)])
        self.reset_parameters()

    def reset_parameters(self):
        for ia3_lff in self.ia3_lff:
            nn.init.constant_(ia3_lff, 1.0)

    def forward(self, x, **kwargs):
        output = x.transpose(-1, 1)
        output = self.conv1(output)
        output = self.dropout(self.activation(output))

        tuner_index = kwargs.get("tuner_index", 0)
        init = kwargs.get("init", False)
        if not init:
            lff = self.ia3_lff[tuner_index]
            output = lff.unsqueeze(-1) * output
        output = self.conv2(output)
        output = self.dropout(output.transpose(-1, 1))
        return output


class EncoderLayerWithIA3(EncoderLayer):
    def __init__(self, attention, d_model, d_ff=None, num_ia3=1, dropout=0.1, activation="relu"):
        super().__init__(attention, d_model, d_ff, dropout, activation)
        self.ffn = FFNLayerIA3(d_model, d_ff, num_ia3, dropout, activation)
