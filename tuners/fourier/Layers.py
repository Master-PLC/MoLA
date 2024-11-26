import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierLinear(nn.Module):
    def __init__(self, base_layer, num_loras, n_frequency=100):
        super().__init__()
        if isinstance(base_layer, (nn.Linear, nn.Conv1d)):
            self.base_layer = base_layer
        else:
            raise ValueError("Base layer must be an instance of nn.Linear or nn.Conv1d")

        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            assert base_layer.kernel_size[0] == 1, "Only 1D convolution is supported"
            self.in_features = base_layer.in_channels
            self.out_features = base_layer.out_channels

        # LoRA parameters
        self.indices = []
        for _ in range(num_loras):
            ind = torch.randperm(self.in_features * self.out_features)[:n_frequency]
            ind = torch.stack([ind // self.in_features, ind % self.out_features], dim=0)
            self.indices.append(ind)
        self.spectrum = nn.ParameterList([nn.Parameter(torch.randn(n_frequency)) for _ in range(num_loras)])

    def forward(self, x, **kwargs):
        init = kwargs.get("init", False)
        if init:
            return self.base_layer(x)

        tuner_index = kwargs.get("tuner_index", 0)

        spectrum = self.spectrum[tuner_index]
        indices = self.indices[tuner_index]
        dense_s = torch.zeros((self.in_features, self.out_features), dtype=spectrum.dtype, device=spectrum.device)
        dense_s[indices[0], indices[1]] = spectrum
        weight_delta = torch.fft.ifft2(dense_s).real.T

        # Apply adjustment to the original weight
        if isinstance(self.base_layer, nn.Linear):
            adjusted_weight = self.base_layer.weight + weight_delta
            return F.linear(x, adjusted_weight, self.base_layer.bias)
        else:
            adjusted_weight = self.base_layer.weight + weight_delta.unsqueeze(-1)
            return F.conv1d(x, adjusted_weight, bias=self.base_layer.bias, stride=self.base_layer.stride, padding=self.base_layer.padding)
