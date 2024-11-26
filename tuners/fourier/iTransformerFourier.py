import re

import torch.nn as nn

from models.iTransformer import Model as iTransformer
from tuners.fourier.Layers import FourierLinear


class Model(iTransformer):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self.num_loras = configs.label_pred_len // configs.pred_len
        target_modules = configs.target_modules or ['encoder.*.ffn']
        self.target_regex = [re.compile(target) for target in target_modules]

        # Temporary list to hold modules that need to be replaced
        replacements = []

        # Collect replacements without modifying the _modules dict
        for name, module in self.named_modules():
            if any(target.search(name) for target in self.target_regex):
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    replacements.append((name, FourierLinear(module, self.num_loras, configs.n_frequency)))

        # Apply the collected replacements
        for name, new_module in replacements:
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)
