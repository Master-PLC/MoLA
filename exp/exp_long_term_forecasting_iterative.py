import os
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
from utils.ot_dist import *
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch, pca_torch)
from utils.tools import (EarlyStopping, adjust_learning_rate, ensure_path,
                         visual)

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Iterative(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Iterative, self).__init__(args)
        self.model.to(self.device)
        self.seq_len = args.seq_len
        self.pred_len = args.label_pred_len
        self.label_len = args.label_len

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        pretrain_model_path = self.args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f"loading pretrain model")
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict)
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

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = []
                for pl in range(self.pred_len):
                    if pl < self.seq_len:
                        _batch_x = torch.cat([batch_x[:, pl:], *outputs], dim=1)
                        _batch_x_mark = torch.cat([batch_x_mark[:, pl:], batch_y_mark[:, self.label_len:self.label_len+pl]], dim=1)
                    else:
                        _batch_x = torch.cat(outputs[-self.seq_len:], dim=1)
                        _batch_x_mark = batch_y_mark[:, pl+self.label_len-self.seq_len:pl+self.label_len, :]
                    
                    # decoder input
                    _dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                    if pl < self.label_len:
                        _dec_inp = torch.cat([batch_y[:, pl:self.label_len, :], *outputs, _dec_inp], dim=1).float().to(self.device)
                    else:
                        _dec_inp = torch.cat([*outputs[-self.label_len:], _dec_inp], dim=1).float().to(self.device)
                    _batch_y_mark = batch_y_mark[:, pl:pl+self.label_len+1, :]
                    # print(i, pl, _batch_x.shape, _dec_inp.shape, _batch_x_mark.shape, _batch_y_mark.shape)

                    # encoder - decoder
                    if self.args.output_attention:
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)[0]
                    else:
                        # outputs shape: [B, P, D]
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        print("Validation cost time: {}".format(time.time() - eval_time))
        total_loss = np.average(total_loss)
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
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = []
                for pl in range(self.pred_len):
                    if pl < self.seq_len:
                        _batch_x = torch.cat([batch_x[:, pl:], *outputs], dim=1)
                        _batch_x_mark = torch.cat([batch_x_mark[:, pl:], batch_y_mark[:, self.label_len:self.label_len+pl]], dim=1)
                    else:
                        _batch_x = torch.cat(outputs[-self.seq_len:], dim=1)
                        _batch_x_mark = batch_y_mark[:, pl+self.label_len-self.seq_len:pl+self.label_len, :]
                    
                    # decoder input
                    _dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                    if pl < self.label_len:
                        _dec_inp = torch.cat([batch_y[:, pl:self.label_len, :], *outputs, _dec_inp], dim=1).float().to(self.device)
                    else:
                        _dec_inp = torch.cat([*outputs[-self.label_len:], _dec_inp], dim=1).float().to(self.device)
                    _batch_y_mark = batch_y_mark[:, pl:pl+self.label_len+1, :]
                    # print(i, pl, _batch_x.shape, _dec_inp.shape, _batch_x_mark.shape, _batch_y_mark.shape)

                    # encoder - decoder
                    if self.args.output_attention:
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)[0]
                    else:
                        # outputs shape: [B, P, D]
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss, self.step)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)
            # self.writer.add_scalar(f'{self.pred_len}/test/loss', test_loss, self.epoch)

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #         epoch + 1, train_steps, train_loss, vali_loss, test_loss
            #     )
            # )
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            pretrain_model_path = self.args.pretrain_model_path
            if pretrain_model_path and os.path.exists(pretrain_model_path):
                state_dict = torch.load(pretrain_model_path)
            else:
                state_dict = torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'))
            self.model.load_state_dict(state_dict)
            print(f"loading model successfully")

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        ensure_path(folder_path)

        self.model.eval()
        metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = []
                for pl in range(self.pred_len):
                    if pl < self.seq_len:
                        _batch_x = torch.cat([batch_x[:, pl:], *outputs], dim=1)
                        _batch_x_mark = torch.cat([batch_x_mark[:, pl:], batch_y_mark[:, self.label_len:self.label_len+pl]], dim=1)
                    else:
                        _batch_x = torch.cat(outputs[-self.seq_len:], dim=1)
                        _batch_x_mark = batch_y_mark[:, pl+self.label_len-self.seq_len:pl+self.label_len, :]
                    # decoder input
                    _dec_inp = torch.zeros_like(batch_y[:, -1:, :]).float()
                    if pl < self.label_len:
                        _dec_inp = torch.cat([batch_y[:, pl:self.label_len, :], *outputs, _dec_inp], dim=1).float().to(self.device)
                    else:
                        _dec_inp = torch.cat([*outputs[-self.label_len:], _dec_inp], dim=1).float().to(self.device)
                    _batch_y_mark = batch_y_mark[:, pl:pl+self.label_len+1, :]
                    # print(i, pl, _batch_x.shape, _dec_inp.shape, _batch_x_mark.shape, _batch_y_mark.shape)

                    # encoder - decoder
                    if self.args.output_attention:
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)[0]
                    else:
                        # outputs shape: [B, P, D]
                        output = self.model(_batch_x, _batch_x_mark, _dec_inp, _batch_y_mark)
                    outputs.append(output)
                outputs = torch.cat(outputs, dim=1)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, :]
                batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
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

        # result save
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        m = metric_collector.compute()
        mae, mse, rmse, mape, mspe = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"]
        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.close()

        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))
        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs)
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)

        return
