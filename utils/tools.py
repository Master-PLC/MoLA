import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, state_mark=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.state_mark = state_mark

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        state_dict = model.state_dict()
        if self.state_mark is not None:
            state_dict = {k: v for k, v in state_dict.items() if self.state_mark in k}
            save_path = os.path.join(path, f"{self.state_mark}_checkpoint.pth")
        else:
            save_path = os.path.join(path, "checkpoint.pth")
        torch.save(state_dict, save_path)
        self.val_loss_min = val_loss


class EarlyStoppingManager:
    def __init__(self, num_candi=1, patience=7, verbose=False, delta=0, state_mark=None):
        self.num_candi = num_candi
        self.patience = patience
        self.verbose = verbose
        self.counter = [0] * num_candi
        self.best_score = [None] * num_candi
        self.early_stop = [False] * num_candi
        self.val_loss_min = [np.Inf] * num_candi
        self.delta = delta
        self.state_mark = state_mark

    def __call__(self, val_loss, model, path):
        save = False
        for i in range(self.num_candi):
            if self.early_stop[i]:
                continue

            _val_loss = val_loss[i]
            score = -_val_loss
            if self.best_score[i] is None:
                self.best_score[i] = score
                print(f'Candidate {i} | Validation loss decreased ({self.val_loss_min[i]:.6f} --> {_val_loss:.6f}).')
                self.val_loss_min[i] = _val_loss
                save = True
            elif score < self.best_score[i] + self.delta:
                self.counter[i] += 1
                print(f'Candidate {i} | EarlyStopping counter: {self.counter[i]} out of {self.patience}')
                if self.counter[i] >= self.patience:
                    self.early_stop[i] = True
                    print(f'Candidate {i} | Early stopping')
            else:
                self.best_score[i] = score
                print(f'Candidate {i} | Validation loss decreased ({self.val_loss_min[i]:.6f} --> {_val_loss:.6f}).')
                self.val_loss_min[i] = _val_loss
                self.counter[i] = 0
                save = True

        if save:
            self.save_checkpoint(model, path)

    def save_checkpoint(self, model, path):
        state_dict = model.state_dict()
        if self.state_mark is not None:
            state_dict = {k: v for k, v in state_dict.items() if self.state_mark in k}
            save_path = os.path.join(path, f"{self.state_mark}_checkpoint.pth")
        else:
            save_path = os.path.join(path, "checkpoint.pth")
        torch.save(state_dict, save_path)

    @property
    def early_stop_all(self):
        return all(self.early_stop)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class EvalAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            values = eval(values)
        except:
            try:
                values = eval(values.lower().capitalize())
            except:
                pass
        setattr(namespace, self.dest, values)


def get_shared_parameters(model, tag="shared"):
    if hasattr(model, "shared_parameters"):
        return model.shared_parameters()

    shared_parameters = []
    for name, param in model.named_parameters():
        if tag in name:
            shared_parameters.append(param)
    if len(shared_parameters) == 0:
        return model.parameters()
    return shared_parameters


def get_task_specific_parameters(model, tag="task_specific"):
    if hasattr(model, "task_specific_parameters"):
        return model.task_specific_parameters()

    task_specific_parameters = []
    for name, param in model.named_parameters():
        if tag in name:
            task_specific_parameters.append(param)
    if len(task_specific_parameters) == 0:
        return []
    return task_specific_parameters


def get_nb_trainable_parameters_info(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"\ntrainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}\n")


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


class PParameter(nn.Parameter):
    def __repr__(self):
        tensor_type = str(self.data.type()).split('.')[-1]
        size_str = " x ".join(map(str, self.shape))
        return f"Parameter containing: [{tensor_type} of size {size_str}]"
