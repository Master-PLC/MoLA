import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .iTransformer import Model as iTransformer


class Model(iTransformer):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super().__init__(configs)

        # Decoder
        if 'long_term_forecast' in self.task_name or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, **kwargs)

        dec_out = self.projection(enc_out) if isinstance(self.projection, nn.Linear) else self.projection(enc_out, **kwargs)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out[:, :1]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, 1, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, 1, 1))
        return dec_out
