import re

import torch.nn as nn

from models.Autoformer import Model as Autoformer
from tuners.lora.Layers import LoraConv1d, LoraLinear


class Model(Autoformer):
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
