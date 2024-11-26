import re

import torch
import torch.nn as nn

from models.iTransformer import Model as iTransformer
from tuners.lora.Layers import LoraConv1d, LoraLinear


class Model(iTransformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_loras = configs.label_pred_len // configs.pred_len
        target_modules = configs.target_modules or ['encoder.*.ffn']
        self.target_regex = [re.compile(target) for target in target_modules]

        lora_configs = {
            'init_type': configs.init_type,
            'std_scale': configs.std_scale,
            'apply_bias': configs.apply_bias
        }

        # Temporary list to hold modules that need to be replaced
        replacements = []
        # Collect replacements without modifying the _modules dict
        for name, module in self.named_modules():
            if any(target.search(name) for target in self.target_regex):
                if isinstance(module, nn.Linear):
                    replacements.append((name, LoraLinear(module, self.num_loras, configs.r, **lora_configs)))
                elif isinstance(module, nn.Conv1d):
                    replacements.append((name, LoraConv1d(module, self.num_loras, configs.r, **lora_configs)))

        # Apply the collected replacements
        for name, new_module in replacements:
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)

        # Decoder
        if 'long_term_forecast' in self.task_name or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.label_pred_len, bias=True)

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

        tuner_index = kwargs.get('tuner_index', 0)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out[:, tuner_index*self.pred_len:(tuner_index+1)*self.pred_len]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out