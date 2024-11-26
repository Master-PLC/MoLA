import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(self, base_layer, num_loras, r, init_type="kaiming", std_scale=0.001, apply_bias=False):
        super().__init__()
        if isinstance(base_layer, nn.Linear):
            self.base_layer = base_layer
        else:
            raise ValueError("Base layer must be an instance of nn.Linear")

        self.apply_bias = apply_bias
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # LoRA parameters
        self.r = r 
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.randn(self.r, self.in_features)) for _ in range(num_loras)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.out_features, self.r)) for _ in range(num_loras)
        ])
        if self.apply_bias:
            self.lora_bias = nn.ParameterList([
                nn.Parameter(torch.randn(self.out_features)) for _ in range(num_loras)
            ])

        self.reset_parameters(init_type, std_scale)

    def reset_parameters(self, init_type, std_scale):
        for lora_A, lora_B in zip(self.lora_A, self.lora_B):
            if init_type == "kaiming":
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            elif init_type == "normal":
                nn.init.normal_(lora_A, std=std_scale)
            nn.init.zeros_(lora_B)
        
        if self.apply_bias:
            for lora_bias in self.lora_bias:
                nn.init.zeros_(lora_bias)

    def forward(self, x, **kwargs):
        init = kwargs.get("init", False)
        if init:
            return self.base_layer(x)

        tuner_index = kwargs.get("tuner_index", 0)
        lora_A = self.lora_A[tuner_index]
        lora_B = self.lora_B[tuner_index]
        if self.apply_bias:
            lora_bias = self.lora_bias[tuner_index]

        # Calculate the low-rank adjustment to weights independently
        weight_delta = lora_B @ lora_A
        # Apply adjustment to the original weight
        adjusted_weight = self.base_layer.weight + weight_delta
        bias = self.base_layer.bias if not self.apply_bias else self.base_layer.bias + lora_bias
        return F.linear(x, adjusted_weight, bias)


class LoraConv1d(nn.Module):
    def __init__(self, base_layer, num_loras, r, init_type="kaiming", std_scale=0.01, apply_bias=False):
        super().__init__()
        if isinstance(base_layer, nn.Conv1d):
            self.base_layer = base_layer
        else:
            raise ValueError("Base layer must be an instance of nn.Conv1d")

        self.apply_bias = apply_bias
        self.in_channels = base_layer.in_channels
        self.out_channels = base_layer.out_channels
        self.kernel_size = base_layer.kernel_size[0]  # Assuming kernel_size is a tuple

        # LoRA parameters
        self.r = r
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.randn(self.r, self.in_channels * self.kernel_size)) for _ in range(num_loras)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.randn(self.out_channels, self.r)) for _ in range(num_loras)
        ])
        if self.apply_bias:
            self.lora_bias = nn.ParameterList([
                nn.Parameter(torch.randn(self.out_channels)) for _ in range(num_loras)
            ])
        self.reset_parameters(init_type, std_scale)

    def reset_parameters(self, init_type, std_scale):
        for lora_A, lora_B in zip(self.lora_A, self.lora_B):
            if init_type == "kaiming":
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            elif init_type == "normal":
                nn.init.normal_(lora_A, std=std_scale)
            nn.init.zeros_(lora_B)

        if self.apply_bias:
            for lora_bias in self.lora_bias:
                nn.init.zeros_(lora_bias)

    def forward(self, x, **kwargs):
        init = kwargs.get("init", False)
        if init:
            return self.base_layer(x)

        tuner_index = kwargs.get("tuner_index", 0)
        lora_A = self.lora_A[tuner_index]
        lora_B = self.lora_B[tuner_index]
        if self.apply_bias:
            lora_bias = self.lora_bias[tuner_index]

        # Low-rank transformation of weights
        weight_delta = torch.matmul(lora_B, lora_A)
        weight_delta = weight_delta.view(self.out_channels, self.in_channels, self.kernel_size)

        # Adjust original weights with the low-rank approximation
        adjusted_weights = self.base_layer.weight + weight_delta
        bias = self.base_layer.bias if not self.apply_bias else self.base_layer.bias + lora_bias
        # Perform the convolution using the adjusted weights
        return F.conv1d(x, adjusted_weights, bias=bias, stride=self.base_layer.stride, padding=self.base_layer.padding)
