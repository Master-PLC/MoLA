import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import DecoderLayer, EncoderLayer


class FFNLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu") -> None:
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, **kwargs):
        if self.conv1.weight.device != x.device:
            self.conv1 = self.conv1.to(x.device)
            self.conv2 = self.conv2.to(x.device)
        output = x.transpose(-1, 1)
        output = self.conv1(output) if isinstance(self.conv1, nn.Conv1d) else self.conv1(output, **kwargs)
        output = self.dropout(self.activation(output))
        output = self.conv2(output) if isinstance(self.conv2, nn.Conv1d) else self.conv2(output, **kwargs)
        output = self.dropout(output.transpose(-1, 1))
        return output


class EncoderLayerWithAdapter(EncoderLayer):
    def __init__(self, attention, d_model, d_ff=None, num_group=1, num_adapters=1, r=None, dropout=0.1, activation="relu"):
        super(EncoderLayerWithAdapter, self).__init__(attention, d_model, d_ff, dropout, activation)
        self.num_adapters = num_adapters
        r = r or d_model // 4
        self.adapter_score = nn.ParameterList(
            [nn.Parameter(torch.randn(num_adapters)) for _ in range(num_group)]
        )
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
            adapter_score = self.adapter_score[tuner_index]
            adapter_score = F.softmax(adapter_score)

            adapter_output = []
            for i in range(self.num_adapters):
                adapter_output.append(self.adapter[i](y))
            adapter_output = torch.stack(adapter_output, dim=0)
            adapter_output = torch.einsum('n,nbtd->btd', adapter_score, adapter_output)
            y = y + adapter_output

        return self.norm2(x + y), attn


class DecoderLayerWithAdapter(DecoderLayer):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, num_group=1, num_adapters=1, r=None, dropout=0.1, activation="relu"):
        super(DecoderLayerWithAdapter, self).__init__(self_attention, cross_attention, d_model, d_ff, dropout, activation)
        self.num_adapters = num_adapters
        r = r or d_model // 4
        self.adapter_score = nn.ParameterList(
            [nn.Parameter(torch.randn(num_adapters)) for _ in range(num_group)]
        )
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
            adapter_score = self.adapter_score[tuner_index]
            adapter_score = F.softmax(adapter_score)
            
            adapter_output = []
            for i in range(self.num_adapters):
                adapter_output.append(self.adapter[i](y))
            adapter_output = torch.stack(adapter_output, dim=0)
            adapter_output = torch.einsum('n,nbtd->btd', adapter_score, adapter_output)
            y = y + adapter_output

        return self.norm3(x + y)
