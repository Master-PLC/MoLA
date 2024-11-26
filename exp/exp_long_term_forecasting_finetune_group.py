import os
import re
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import (EarlyStoppingManager, adjust_learning_rate,
                         ensure_path, get_nb_trainable_parameters_info,
                         split_list, visual)

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Finetune_Group(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Finetune_Group, self).__init__(args)
        self.model_pred_len = args.pred_len
        self.pred_len = args.label_pred_len
        self.label_len = args.label_len
        self.tuner_type = args.tuner_type
        self.num_tuner = self.pred_len // self.model_pred_len
        self.num_tuner_online = args.num_tuner_online

        self._mark_tuner_as_trainable(tuner_indexes=0, init=True)
        get_nb_trainable_parameters_info(self.model)

    def _mark_tuner_as_trainable(self, tuner_indexes=[0], init=False):
        tuner_indexes = [str(tuner_indexes)] if isinstance(tuner_indexes, int) else list(map(str, tuner_indexes))
        tuner_indexes = "|".join(tuner_indexes)
        target = f"{self.tuner_type}[^.]*\.({tuner_indexes})"
        other_target = f"{self.tuner_type}[^.]*\.(\d+)"

        for n, p in self.model.named_parameters():
            if self.tuner_type not in n:  ## base parameters
                p.requires_grad = False
                if init:
                    p.data = p.data.to(self.device)
            elif not re.search(target, n):  ## other tuners
                if re.search(other_target, n):
                    p.requires_grad = False
                    p.data = p.data.to('cpu')
                else:
                    p.requires_grad = True  ## MoLora's lora_A and lora_B
                    p.data = p.data.to(self.device)
            else:  ## tuner to train
                p.requires_grad = True
                p.data = p.data.to(self.device)

            p.grad = None  ## avoid gradient accumulation

        if init:
            for buf in self.model.buffers():
                buf.data = buf.data.to(self.device)

    def _tuner_to_gpu(self, tuner_indexes=0):
        tuner_indexes = [str(tuner_indexes)] if isinstance(tuner_indexes, int) else list(map(str, tuner_indexes))
        tuner_indexes = "|".join(tuner_indexes)
        target = f"{self.tuner_type}[^.]*\.({tuner_indexes})"
        other_target = f"{self.tuner_type}[^.]*\.(\d+)"
        for n, p in self.model.named_parameters():
            if self.tuner_type not in n:  ## base parameters
                pass
            elif not re.search(target, n):  ## other tuners
                if re.search(other_target, n):
                    p.data = p.data.to('cpu')
                else:
                    p.data = p.data.to(self.device)
            else:  ## tuner to train
                p.data = p.data.to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        pretrain_model_path = self.args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f"loading pretrain model")
            state_dict = torch.load(pretrain_model_path, map_location=self.device)
            new_state_dict = {}
            target_modules = self.args.target_modules  # ['encoder.*.ffn']
            target_regex = [re.compile(target) for target in target_modules]
            for name, param in state_dict.items():
                new_name = name
                # 检查键名是否匹配 target_regex 中的任何一个模式
                if any(regex.search(name) for regex in target_regex):
                    # 如果键名以 'weight' 或 'bias' 结尾，插入 'base_layer'
                    if name.endswith('weight') or name.endswith('bias'):
                        parts = name.split('.')
                        # 在最后一个元素前插入 'base_layer'
                        new_name = '.'.join(parts[:-1] + ['base_layer', parts[-1]])
                new_state_dict[new_name] = param
            model.load_state_dict(new_state_dict, strict=False)
            print(f"loading pretrain model successfully")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # def _select_optimizer(self):
    #     ### N optimizer for N tuner
    #     param_groups = []
    #     for tuner_index in range(self.pred_len):
    #         target = f"{self.tuner_type}[^.]*\.{tuner_index}"
    #         target_regex = re.compile(target)
    #         params = [p for n, p in self.model.named_parameters() if target_regex.search(n)]
    #         param_groups.append({"params": params, 'lr': self.args.learning_rate})
    #     model_optim = optim.Adam(param_groups)
    #     return model_optim

    def _select_criterion(self, reduction='mean'):
        criterion = nn.MSELoss(reduction=reduction)
        return criterion

    def vali(self, vali_data, vali_loader, criterion, init=False):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        tuner_list = list(range(self.num_tuner))
        tuner_list = split_list(tuner_list, self.num_tuner_online)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = []
                for tuners in tuner_list:
                    self._tuner_to_gpu(tuners)

                    for ti in tuners:
                        kwargs = {
                            "tuner_index": ti,
                            "init": init
                        }

                        _batch_y_mark = torch.cat([
                            batch_y_mark[:, :self.label_len, :], 
                            batch_y_mark[:, self.label_len+ti*self.model_pred_len:self.label_len+(ti+1)*self.model_pred_len]
                        ], dim=1)

                        # decoder input
                        _dec_inp = torch.zeros_like(batch_y[:, -self.model_pred_len:, :]).float()
                        _dec_inp = torch.cat([batch_y[:, :self.label_len, :], _dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        if self.args.output_attention:
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)[0]
                        else:
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)
                        outputs.append(output)
                outputs = torch.cat(outputs, dim=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true).mean(dim=(0, -1))
                loss = loss.view(-1, self.model_pred_len).mean(dim=1)

                total_loss.append(loss)

        print("Validation cost time: {}".format(time.time() - eval_time))
        total_loss = torch.stack(total_loss).mean(dim=0).cpu().numpy()
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        ensure_path(path)
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStoppingManager(
            num_candi=self.num_tuner, patience=self.args.patience, verbose=True, state_mark=self.tuner_type
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(reduction='none')

        ## init evaluation
        _criterion = self._select_criterion(reduction='none')
        vali_loss = self.vali(vali_data, vali_loader, _criterion, init=True)
        # test_loss = self.vali(test_data, test_loader, _criterion, init=True)
        for ti in range(self.num_tuner):
            self.writer.add_scalar(f'{self.pred_len}/vali/{ti}/loss', vali_loss[ti], 0)
            # self.writer.add_scalar(f'{self.pred_len}/test/{ti}/loss', test_loss[ti], 0)
            # print(f"Tuner: {ti}, Epoch: 0, Steps: 0 | Vali Loss: {vali_loss[ti]:.7f} Test Loss: {test_loss[ti]:.7f}")
            print(f"Tuner: {ti}, Epoch: 0, Steps: 0 | Vali Loss: {vali_loss[ti]:.7f}")
        early_stopping(vali_loss, self.model, path)
        # print('Global vali loss:', vali_loss.mean(), 'Global test loss:', test_loss.mean())
        print('Global vali loss:', vali_loss.mean())
        ## init evaluation

        tuner_list = list(range(self.num_tuner))
        tuner_list = split_list(tuner_list, self.num_tuner_online)
        num_tuner_group = len(tuner_list)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            train_loss = {f'{ti}': [] for ti in range(self.num_tuner)}
            self.model.train()

            epoch_time = time.time()

            last_tuners = []
            for j, tuners in enumerate(tuner_list):
                _tuners = [ti for ti in tuners if not early_stopping.early_stop[ti]]
                _tuners = last_tuners + _tuners
                if len(_tuners) >= self.num_tuner_online:
                    last_tuners = _tuners[self.num_tuner_online:]
                    _tuners = _tuners[:self.num_tuner_online]
                else:
                    if j < num_tuner_group - 1:
                        last_tuners = _tuners
                        continue

                if len(_tuners) == 0:
                    break

                tuner_time = time.time()
                iter_count = 0
                step_count = 0

                self._mark_tuner_as_trainable(_tuners)

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    step_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs = []
                    for ti in _tuners:
                        kwargs = {"tuner_index": ti}

                        ### lora 跟 base 的decoder区分
                        _batch_y_mark = torch.cat([
                            batch_y_mark[:, :self.label_len, :], 
                            batch_y_mark[:, self.label_len+ti*self.model_pred_len:self.label_len+(ti+1)*self.model_pred_len]
                        ], dim=1)

                        # decoder input
                        _dec_inp = torch.zeros_like(batch_y[:, -self.model_pred_len:, :]).float()
                        _dec_inp = torch.cat([batch_y[:, :self.label_len, :], _dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        if self.args.output_attention:
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)[0]
                        else:
                            # outputs shape: [B, P, D]
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)
                        outputs.append(output)
                    outputs = torch.cat(outputs, dim=1)  # [B, num_tuner_online, D]

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                    _points = [_ti for ti in _tuners for _ti in range(ti*self.model_pred_len, (ti+1)*self.model_pred_len)]
                    batch_y = batch_y[:, _points]

                    loss = criterion(outputs, batch_y)
                    loss = loss.mean(dim=(0, -1))  # k * model_pred_len
                    loss = loss.view(-1, self.model_pred_len).mean(dim=1)  # k

                    loss_ = loss.clone().detach().cpu().numpy()
                    for ti, l in zip(_tuners, loss_):
                        train_loss[f'{ti}'].append(l)
                    loss = loss.sum()

                    loss.backward()
                    model_optim.step()

                    if (i + 1) % 100 == 0:
                        print(f"\tTuner group: {j}, iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                        cost_time = time.time() - time_now
                        speed = cost_time / iter_count
                        left_time = speed * (((self.args.train_epochs - epoch) * num_tuner_group - j) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                        iter_count = 0
                        time_now = time.time()

                tuner_str = "|".join(map(str, _tuners))
                print(f"Tuners: {tuner_str}, Epoch: {epoch+1}, cost time: {time.time() - tuner_time}")

            _criterion = self._select_criterion(reduction='none')
            vali_loss = self.vali(vali_data, vali_loader, _criterion)
            # test_loss = self.vali(test_data, test_loader, _criterion)

            for ti in range(self.num_tuner):
                train_loss_ti = np.average(train_loss[f'{ti}'])
                self.writer.add_scalar(f'{self.pred_len}/train/{ti}/loss', train_loss_ti, self.epoch)
                self.writer.add_scalar(f'{self.pred_len}/vali/{ti}/loss', vali_loss[ti], self.epoch)
                # self.writer.add_scalar(f'{self.pred_len}/test/{ti}/loss', test_loss[ti], self.epoch)
                print(
                    f"Tuner: {ti}, Epoch: {epoch+1}, Steps: {train_steps} | "
                    f"Train Loss: {train_loss_ti:.7f} Vali Loss: {vali_loss[ti]:.7f}"
                    # f"Test Loss: {test_loss[ti]:.7f}"
                )

            early_stopping(vali_loss, self.model, path)

            self.step += step_count
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            if early_stopping.early_stop_all:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, f'{self.tuner_type}_checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            best_model_path = os.path.join(self.args.checkpoints, setting, f'{self.tuner_type}_checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)
            print('loading model successfully')

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        ensure_path(folder_path)
        # result save
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)

        self.model.eval()
        tuner_list = list(range(self.num_tuner))
        _tuner_list = split_list(tuner_list, self.num_tuner_online)

        metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = []
                for tuners in _tuner_list:
                    self._tuner_to_gpu(tuners)
                    for ti in tuners:
                        kwargs = {"tuner_index": ti}

                        _batch_y_mark = torch.cat([
                            batch_y_mark[:, :self.label_len, :], 
                            batch_y_mark[:, self.label_len+ti*self.model_pred_len:self.label_len+(ti+1)*self.model_pred_len]
                        ], dim=1)

                        # decoder input
                        _dec_inp = torch.zeros_like(batch_y[:, -self.model_pred_len:, :]).float()
                        _dec_inp = torch.cat([batch_y[:, :self.label_len, :], _dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        if self.args.output_attention:
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)[0]
                        else:
                            output = self.model(batch_x, batch_x_mark, _dec_inp, _batch_y_mark, **kwargs)
                        outputs.append(output)
                outputs = torch.cat(outputs, dim=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach()
                batch_y = batch_y.detach()
                if test_data.scale and self.args.inverse:
                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()

                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                    outputs = torch.from_numpy(outputs).to(self.device)
                    batch_y = torch.from_numpy(batch_y).to(self.device)

                outputs = outputs[:, :, f_dim:].contiguous()
                batch_y = batch_y[:, :, f_dim:].contiguous()

                metric_collector.update(outputs, batch_y)

                if self.output_pred or self.output_vis:
                    inp = batch_x.cpu().numpy()
                    pred = outputs.cpu().numpy()
                    true = batch_y.cpu().numpy()

                    inputs.append(inp)
                    preds.append(pred)
                    trues.append(true)

                if i % 20 == 0 and self.output_vis:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.output_pred:
            inputs = np.array(inputs)
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

        m = metric_collector.compute()
        mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        if self.writer is None:
            self.writer = self._create_writer(res_path)
        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs)
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)

        return
