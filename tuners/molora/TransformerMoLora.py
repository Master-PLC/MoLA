import re

import torch.nn as nn

from models.Transformer import Model as Transformer
from tuners.molora.Layers import LoraConv1d, LoraLinear


class Model(Transformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_group = configs.label_pred_len // configs.pred_len
        self.num_loras = configs.n_exp
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
                    replacements.append((name, LoraLinear(module, self.num_group, self.num_loras, configs.r, **lora_configs)))
                elif isinstance(module, nn.Conv1d):
                    replacements.append((name, LoraConv1d(module, self.num_group, self.num_loras, configs.r, **lora_configs)))

        # Apply the collected replacements
        for name, new_module in replacements:
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
