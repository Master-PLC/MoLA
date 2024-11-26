import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from layers.FMLP_Blocks import FFNBlock, FilterMLPBlock


class MLPLayer(nn.Module):
    def __init__(self, d_model, d_hidden, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(d_model, d_hidden)
        self.activation = activation

    def forward(self, input):
        output = self.linear(input)
        if self.activation is not None:
            output = self.activation(output)

        return output


class MLPBlock(nn.Module):
    def __init__(self, d_model, d_hidden, n_layer, activation=F.relu, **kwconfigs):
        super().__init__()
        layers = []
        for l in range(n_layer):
            d_m = d_model if l == 0 else d_hidden
            _activation = activation if l < n_layer - 1 else None
            layers.append(MLPLayer(d_m, d_hidden, _activation))
        self.MLP = nn.ModuleList(layers)

    def forward(self, input):
        output = input
        for mlp in self.MLP:
            output = mlp(output)
        return output


class Tower(nn.Module):
    def __init__(self, hidden, c_out, n_layer=1, activation=F.relu):
        super().__init__()

        layers = []
        for l in range(n_layer):
            d_h = hidden if l < n_layer - 1 else c_out
            _activation = activation if l < n_layer - 1 else None
            layers.append(MLPLayer(hidden, d_h, _activation))
        self.Tower = nn.ModuleList(layers)

    def forward(self, x):
        output = x
        for mlp in self.Tower:
            output = mlp(output)
        return output


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class ReduceRNNOutput(nn.Module):
    def forward(self, x):
        return x[0]


class FMLPBlock(nn.Module):

    def __init__(self, x_dim, hidden, round, seq_len, dropout, init_ratio):

        super().__init__()

        self.seq_len = seq_len
        self.x_dim = x_dim
        self.round = round
        self.dropout = dropout
        self.hidden = hidden

        self.init_ratio = init_ratio

        self.TA = nn.ModuleList([FilterMLPBlock(self.hidden, dropout=self.dropout, seq_len=self.seq_len, init_ratio=self.init_ratio) for i in range(self.round)])
        self.ffn = nn.ModuleList([FFNBlock(self.hidden, self.hidden, dropout=self.dropout) for i in range(self.round)])
        self.mlp1 = nn.Linear(self.x_dim, self.hidden)

    def forward(self, input_tensor):

        output = self.mlp1(input_tensor)
        for round in range(self.round):
            output = self.TA[round](output)
            output = self.ffn[round](output)

        return output


class Experts(nn.Module):

    def __init__(self, configs, c_in, seq_len, hidden, n_layer, n_exp, exp_type='mlp'):
        super().__init__()

        self.n_exp = n_exp

        if exp_type == 'mlp':
            experts = [MLPBlock(c_in, hidden, n_layer) for _ in range(n_exp)]
        elif exp_type == 'lstm':
            experts = [
                nn.Sequential(
                    nn.LSTM(c_in, hidden, n_layer, bidirectional=False, batch_first=True),
                    ReduceRNNOutput()
                ) for _ in range(n_exp)
            ]
        elif exp_type == 'gru':
            experts = [
                nn.Sequential(
                    nn.GRU(c_in, hidden, n_layer, bidirectional=False, batch_first=True),
                    ReduceRNNOutput()
                ) for _ in range(n_exp)
            ]
        elif exp_type == 'conv':
            experts = [
                nn.Sequential(
                    *([Transpose(1, 2), nn.Conv1d(c_in, hidden, 1)] + \
                      [nn.Conv1d(hidden, hidden, 1) for _ in range(n_layer - 1)] + \
                      [Transpose(1, 2)])
                ) for _ in range(n_exp)
            ]
        elif exp_type == 'fmlp':
            experts = [FMLPBlock(c_in, hidden, n_layer, seq_len, configs.dropout, configs.init_ratio) for _ in range(n_exp)]
        else:
            raise NotImplementedError
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        if isinstance(x, list):
            expert_tensors = list(map(lambda exp, input: exp(input), self.experts, x))
        else:
            expert_tensors = list(map(lambda exp: exp(x), self.experts))

        # if isinstance(expert_tensors[0], tuple):
        #     expert_tensors = [exp[0] for exp in expert_tensors]

        expert_tensors = torch.stack(expert_tensors, dim=0)  # (n_exp, batch_size, *, hidden)
        return expert_tensors


class SoftmaxGate(nn.Module):
    def __init__(self, c_in, n_exp):
        super().__init__()
        self.gate = nn.Linear(c_in, n_exp)

    def forward(self, x):
        scores = self.gate(x)
        gate_out = F.softmax(scores, dim=-1)
        return gate_out


class ResidualGate(nn.Module):
    def __init__(self, c_in, n_exp, lambda_r=1.):
        super().__init__()
        self.lambda_r = lambda_r
        self.gate = nn.Linear(c_in, n_exp - 1)

    def forward(self, x):
        scores = self.gate(x)
        scores = F.softmax(scores, dim=-1) * self.lambda_r
        gate_out = torch.ones((*x.shape[:-1], 1)).to(x.device)
        gate_out = torch.cat((gate_out, scores), dim=-1)
        return gate_out


class TopkGate(nn.Module):
    def __init__(self, c_in, n_exp, topk=1):
        super().__init__()

        self.topk = topk
        self.gate = nn.Linear(c_in, n_exp)

    def forward(self, x):
        scores = self.gate(x)
        scores = F.softmax(scores, dim=-1)
        values, indices = torch.topk(scores, self.topk, dim=-1)
        gate_out = torch.zeros_like(scores)
        gate_out.scatter_(dim=-1, index=indices, src=values)
        return gate_out


class Gates(nn.Module):

    def __init__(self, configs, c_in, n_exp, n_gate, seq_len, gate_type='softmax'):
        super().__init__()
        self.n_gate = n_gate
        self.gate_type = gate_type

        if gate_type == 'softmax':
            gates = [SoftmaxGate(c_in, n_exp) for _ in range(n_gate)]
        elif gate_type == 'res':
            gates = [ResidualGate(c_in, n_exp, lambda_r=configs.lambda_r) for _ in range(n_gate)]
        elif gate_type == 'topk':
            gates = [TopkGate(c_in, n_exp, topk=configs.topk) for _ in range(n_gate)]
        elif gate_type == 'learn':
            self.gate_scores = [nn.Parameter(torch.randn(n_exp)) for _ in range(n_gate)]
            [init.kaiming_uniform_(score, a=np.sqrt(5)) for score in self.gate_scores]
        elif gate_type == 'learn_acc_full':
            self.gate_scores = nn.Parameter(torch.randn(n_gate, n_exp, seq_len))
            init.kaiming_uniform_(self.gate_scores, a=np.sqrt(5))
        elif gate_type == 'learn_acc':
            self.gate_scores = nn.Parameter(torch.randn(n_gate, n_exp))
            init.kaiming_uniform_(self.gate_scores, a=np.sqrt(5))
        else:
            raise NotImplementedError

        if 'learn' not in gate_type:
            self.gates = nn.ModuleList(gates)

    def forward(self, x, expert_tensors):
        if 'learn' not in self.gate_type:
            if isinstance(x, list):
                gate_out = list(map(
                    lambda gate, input: torch.einsum('bse,ebsd->bsd', gate(input), expert_tensors), 
                    self.gates, x
                ))  # [(batch_size, *, n_exp) * n_task]
            else:
                gate_out = list(map(
                    lambda gate: torch.einsum('bse,ebsd->bsd', gate(x), expert_tensors), 
                    self.gates
                ))
            gate_out = torch.stack(gate_out, dim=0)
        elif self.gate_type == 'learn':
            gate_out = list(map(
                lambda score: torch.einsum('e,ebsd->bsd', F.softmax(score, dim=-1), expert_tensors), 
                self.gate_scores
            ))
            gate_out = torch.stack(gate_out, dim=0)  # n_gate, batch_size, seq_len, d_model
        elif self.gate_type == 'learn_acc_full':
            gate_scores = F.softmax(self.gate_scores, dim=1)
            gate_scores = F.softmax(gate_scores, dim=2)
            gate_out = torch.einsum('ges,ebsd->gbd', gate_scores, expert_tensors)
        elif self.gate_type == 'learn_acc':
            gate_scores = F.softmax(self.gate_scores, dim=1)
            gate_out = torch.einsum('ge,ebsd->gbsd', gate_scores, expert_tensors)

        return gate_out


class OutputBlock(nn.Module):

    def __init__(self, d_model, d_out=1, n_layer=1, output_type='classification', dropout=0.):
        super().__init__()
        self.output_type = output_type
        self.d_out = d_out

        if output_type == 'classification':
            self.mlp = nn.Linear(d_model, 1)

        elif output_type == 'moe':
            self.mlp = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(d_out)])

        elif output_type == 'mtmd_matrix':
            self.mlp_w = nn.Parameter(torch.randn(d_out, d_model))
            self.mlp_b = nn.Parameter(torch.randn(d_out))

            init.kaiming_uniform_(self.mlp_w, a=np.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.mlp_w)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.mlp_b, a=-bound, b=bound)

        elif output_type == 'tsf':
            self.mlp = nn.Linear(d_model, d_out)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.output_type == "moe":
            output = list(map(
                lambda mlp, _x: mlp(self.dropout(_x[:, -1, :] if _x.ndim == 3 else _x)), 
                self.mlp, x
            ))
            output = torch.concat(output, dim=1)

        elif self.output_type == 'mtmd_matrix':
            # output = x[:, :, -1, :]  # (d_out=T*D, batch_size, d_model)
            output = x if x.ndim == 3 else x[:, :, -1, :]
            output = self.dropout(output)
            output = torch.einsum('od,obd->bo', self.mlp_w, output)  # batch_size, d_out
            output = output + self.mlp_b

        elif self.output_type == 'tsf':
            output = self.mlp(x)
            output = output.transpose(2, 1)

        else:
            output = self.mlp(output)

        if self.output_type == 'classification':
            output = torch.sigmoid(output)
        return output


class MMoEDecoder(nn.Module):
    """Multi-gate Mixture-of-Experts.
    """

    def __init__(self, configs, c_in=None, seq_len=None):
        super().__init__()

        self.configs = configs

        self.c_in = c_in if c_in is not None else configs.enc_in
        self.seq_len = seq_len if seq_len is not None else configs.seq_len

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.n_task = self.pred_len * self.c_out

        self.exp_type = configs.exp_type
        self.gate_type = configs.gate_type
        self.exp_hidden = configs.exp_hidden
        self.n_exp = configs.n_exp
        self.n_exp_layer = configs.exp_layer
        self.n_tower_layer = configs.tower_layer

        self.output_type = configs.output_type

        self.expert_shared = Experts(
            configs, self.c_in, self.seq_len, self.exp_hidden, self.n_exp_layer, self.n_exp, exp_type=self.exp_type
        )
        self.gate_task_specific = Gates(
            configs, self.c_in, self.n_exp, n_gate=self.n_task, seq_len=self.seq_len, gate_type=self.gate_type
        )
        self.OutputBlock = OutputBlock(
            self.exp_hidden, d_out=self.n_task, n_layer=self.n_tower_layer, output_type=self.output_type, dropout=configs.dropout
        )

    def forward(self, x):
        # experts shared
        expert_tensors = self.expert_shared(x)  # (n_exp, batch_size, *, hidden)
        output = self.gate_task_specific(x, expert_tensors)  # [(batch_size, *, hidden) * n_task]
        output = self.OutputBlock(output)
        output = output.reshape(-1, self.pred_len, self.c_out)
        return output

    def shared_parameters(self):
        return self.expert_shared.parameters()

    def task_specific_parameters(self):
        return list(self.gate_task_specific.parameters()) + list(self.OutputBlock.parameters())
