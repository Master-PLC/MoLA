import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    def __init__(self, base_layer, num_group, num_loras, r, init_type="kaiming", std_scale=0.001, apply_bias=False):
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
        self.lora_score = nn.ParameterList([
            nn.Parameter(torch.randn(num_loras)) for _ in range(num_group)
        ])
        self.lora_A = nn.Parameter(torch.randn(num_loras, self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.randn(num_loras, self.out_features, self.r))
        # self.register_buffer("lora_A", lora_A)
        # self.register_buffer("lora_B", lora_B)

        if self.apply_bias:
            self.lora_bias = nn.Parameter(torch.randn(num_loras, self.out_features))
            # self.register_buffer("lora_bias", lora_bias)
        self.reset_parameters(init_type, std_scale)

    def reset_parameters(self, init_type, std_scale):
        if init_type == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif init_type == "normal":
            nn.init.normal_(self.lora_A, std=std_scale)
        nn.init.zeros_(self.lora_B)
        
        if self.apply_bias:
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, **kwargs):
        init = kwargs.get("init", False)
        if init:
            return self.base_layer(x)

        tuner_index = kwargs.get("tuner_index", 0)
        lora_score = self.lora_score[tuner_index]
        lora_score = F.softmax(lora_score)

        # Calculate the low-rank adjustment to weights independently
        weight_delta = torch.eisum('nor,nri->noi', self.lora_B, self.lora_A)  ## num_loras x out_features x in_features
        weight_delta = torch.eisum('n,noi->oi', lora_score, weight_delta)  ## out_features x in_features
        # Apply adjustment to the original weight
        adjusted_weight = self.base_layer.weight + weight_delta
        
        bias = self.base_layer.bias
        if self.apply_bias:
            bias_delta = torch.eisum('n,no->o', lora_score, self.lora_bias)
            bias = bias + bias_delta
        return F.linear(x, adjusted_weight, bias)


class LoraConv1d(nn.Module):
    def __init__(self, base_layer, num_group, num_loras, r, init_type="kaiming", std_scale=0.001, apply_bias=False):
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
        self.lora_score = nn.ParameterList([
            nn.Parameter(torch.randn(num_loras)) for _ in range(num_group)
        ])
        self.lora_A = nn.Parameter(torch.randn(num_loras, self.r, self.in_channels * self.kernel_size))
        self.lora_B = nn.Parameter(torch.randn(num_loras, self.out_channels, self.r))
        # self.register_buffer("lora_A", lora_A)
        # self.register_buffer("lora_B", lora_B)
        
        if self.apply_bias:
            self.lora_bias = nn.Parameter(torch.randn(num_loras, self.out_channels))
            # self.register_buffer("lora_bias", lora_bias)
        self.reset_parameters(init_type, std_scale)

    def reset_parameters(self, init_type, std_scale):
        if init_type == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif init_type == "normal":
            nn.init.normal_(self.lora_A, std=std_scale)
        nn.init.zeros_(self.lora_B)
        
        if self.apply_bias:
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, **kwargs):
        init = kwargs.get("init", False)
        if init:
            return self.base_layer(x)

        tuner_index = kwargs.get("tuner_index", 0)
        lora_score = self.lora_score[tuner_index]
        lora_score = F.softmax(lora_score)

        # Calculate the low-rank adjustment to weights independently
        weight_delta = torch.einsum('nor,nri->noi', self.lora_B, self.lora_A)
        weight_delta = torch.einsum('n,noi->oi', lora_score, weight_delta)
        weight_delta = weight_delta.view(self.out_channels, self.in_channels, self.kernel_size)

        # Adjust original weights with the low-rank approximation
        adjusted_weights = self.base_layer.weight + weight_delta
        
        bias = self.base_layer.bias
        if self.apply_bias:
            bias_delta = torch.einsum('n,no->o', lora_score, self.lora_bias)
            bias = bias + bias_delta

        # Perform the convolution using the adjusted weights
        return F.conv1d(x, adjusted_weights, bias=bias, stride=self.base_layer.stride, padding=self.base_layer.padding)
