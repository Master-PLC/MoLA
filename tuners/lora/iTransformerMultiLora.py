import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class LoraLinear(nn.Module):
    def __init__(self, base_layer, num_loras, r):
        super().__init__()
        if isinstance(base_layer, nn.Linear):
            self.base_layer = base_layer
        else:
            raise ValueError("Base layer must be an instance of nn.Linear")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # LoRA parameters
        self.r = r 
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.r, self.in_features)) for _ in range(num_loras)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.out_features, self.r)) for _ in range(num_loras)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for lora_A, lora_B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

    def forward(self, x, **kwargs):
        tuner_index = kwargs.get("tuner_index", [0])

        outputs = []
        for ti in tuner_index:
            lora_A = self.lora_A[ti]
            lora_B = self.lora_B[ti]

            # Calculate the low-rank adjustment to weights independently
            weight_delta = lora_B @ lora_A
            # Apply adjustment to the original weight
            adjusted_weight = self.base_layer.weight + weight_delta
            bias = self.base_layer.bias
            x_ = x if isinstance(x, torch.Tensor) else x[ti]
            output = F.linear(x_, adjusted_weight, bias)
            outputs.append(output)

        return outputs


class LoraConv1d(nn.Module):
    def __init__(self, base_layer, nume_loras, r):
        super().__init__()
        if isinstance(base_layer, nn.Conv1d):
            self.base_layer = base_layer
        else:
            raise ValueError("Base layer must be an instance of nn.Conv1d")

        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size[0]  # Assuming kernel_size is a tuple

        # LoRA parameters
        self.r = r
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.r, self.in_channels * self.kernel_size)) for _ in range(nume_loras)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.out_channels, self.r)) for _ in range(nume_loras)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for lora_A, lora_B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

    def forward(self, x, **kwargs):
        tuner_index = kwargs.get("tuner_index", [0])

        outputs = []
        for ti in tuner_index:
            lora_A = self.lora_A[ti]
            lora_B = self.lora_B[ti]

            # Low-rank transformation of weights
            weight_delta = torch.matmul(lora_B, lora_A)
            weight_delta = weight_delta.view(self.out_channels, self.in_channels, self.kernel_size)

            # Adjust original weights with the low-rank approximation
            adjusted_weight = self.base_layer.weight + weight_delta

            # Perform the convolution using the adjusted weights
            x_ = x if isinstance(x, torch.Tensor) else x[ti]
            output = F.conv1d(x_, adjusted_weight, bias=self.base_layer.bias, stride=self.base_layer.stride, padding=self.base_layer.padding)
            outputs.append(output)

        return outputs


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
            attns.append(attn)

        if self.norm is not None:
            x = [self.norm(x_) for x_ in x]

        return x, attns


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
        if isinstance(self.conv1, nn.Conv1d):
            output = x.transpose(-1, 1)
            output = self.conv1(output)
            output = self.dropout(self.activation(output))
            output = self.conv2(output)
            output = self.dropout(output.transpose(-1, 1))
        else:
            output = x.transpose(-1, 1) if isinstance(x, torch.Tensor) else [xi.transpose(-1, 1) for xi in x]
            output = self.conv1(output, **kwargs)
            output = [self.dropout(self.activation(out)) for out in output]
            output = self.conv2(output, **kwargs)
            output = [self.dropout(out.transpose(-1, 1)) for out in output]
        return output


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFNLayer(d_model, d_ff, dropout, activation)

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        if isinstance(x, torch.Tensor):
            new_x, attn = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                tau=tau, delta=delta
            )
            x = x + self.dropout(new_x)
            y = x = self.norm1(x)
            y = self.ffn(y, **kwargs)
            output = [self.norm2(x + y_) for y_ in y]

        else:
            xnorm = []
            attn = []
            for xi in x:
                new_x, attn_ = self.attention(
                    xi, xi, xi,
                    attn_mask=attn_mask,
                    tau=tau, delta=delta
                )
                xi = xi + self.dropout(new_x)
                xi = self.norm1(xi)
                xnorm.append(xi)
                attn.append(attn_)

            y = self.ffn(xnorm, **kwargs)
            output = [self.norm2(x_ + y_) for x_, y_ in zip(xnorm, y)]

        return output, attn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Decoder
        if 'long_term_forecast' in self.task_name or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Lora
        self.num_loras = configs.label_pred_len
        target_modules = configs.target_modules or ['encoder.*.ffn']
        self.target_regex = [re.compile(target) for target in target_modules]

        # Temporary list to hold modules that need to be replaced
        replacements = []

        # Collect replacements without modifying the _modules dict
        for name, module in self.named_modules():
            if any(target.search(name) for target in self.target_regex):
                if isinstance(module, nn.Linear):
                    replacements.append((name, LoraLinear(module, self.num_loras, configs.r)))
                elif isinstance(module, nn.Conv1d):
                    replacements.append((name, LoraConv1d(module, self.num_loras, configs.r)))

        # Apply the collected replacements
        for name, new_module in replacements:
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)

        outputs = []
        for enc_out_ in enc_out:
            dec_out = self.projection(enc_out_)
            dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            outputs.append(dec_out)
        return torch.cat(outputs, dim=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        if 'long_term_forecast' in self.task_name or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)
            return dec_out  # [B, L, D]
        else:
            raise ValueError(f"Task name {self.task_name} is not implemented.")
