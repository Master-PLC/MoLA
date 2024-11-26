import argparse
import os
import random
import sys

import numpy as np
import setproctitle
import torch

from exp import EXP_DICT
from utils.print_args import print_args
from utils.tools import EvalAction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--fix_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--rerun', action='store_true', help='rerun', default=False)
    parser.add_argument('--thread', type=int, default=16, help='number of threads')
    parser.add_argument('--active_trial', type=int, default=10, help='active trial')

    # save
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--log_path', type=str, default='./result_long_term_forecast.txt', help='log path')
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--add_noise', action='store_true', help='add noise')
    parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length for model')
    parser.add_argument('--label_pred_len', type=int, default=96, help='prediction sequence length for label')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--reconstruction_type', type=str, default="imputation", help='type of reconstruction')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # FreDF
    parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    parser.add_argument('--rank_ratio', type=float, default=1.0, help='ratio of low rank for PCA')
    parser.add_argument('--pca_dim', type=str, default="all", help="dimension for PCA, choices in ['all','T','D']")
    parser.add_argument('--reinit', type=int, default=0, help="whether reinit for PCA")
    parser.add_argument('--dist_scale', type=float, default=0.1, help="scale factor for ot distance matrix")
    parser.add_argument('--ot_type', type=str, default='emd1d_h', help="type of ot distance, choices in ['emd1d_h']")
    parser.add_argument('--distance', type=str, default="time", help="distance metric for ot")
    parser.add_argument('--normalize', type=int, default=1, help="normalize ot distance matrix")
    parser.add_argument('--reg_sk', type=float, default=0.1, help="strength of entropy regularization in Sinkhorn")

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # for solver
    parser.add_argument('--solver', type=str, default='linear')
    parser.add_argument('--loss_scale', default=None, action=EvalAction)
    parser.add_argument("--iteration_window", default=25, type=int)
    parser.add_argument("--temp", default=2., type=float)
    parser.add_argument("--gamma", default=0.01, type=float)
    parser.add_argument("--w_lr", default=0.025, type=float)
    parser.add_argument("--max_norm", default=1., type=float)
    parser.add_argument("--alpha", default=1.5, type=float)
    parser.add_argument("--params", type=str, default="shared")
    parser.add_argument("--normalization", type=str, default="loss+")
    parser.add_argument("--optim_niter", default=20, type=int)
    parser.add_argument("--update_weights_every", default=1, type=int)
    parser.add_argument("--cmin", type=float, default=0.2)
    parser.add_argument("--c", default=0.4, type=float)
    parser.add_argument("--rescale", default=1, type=int)
    parser.add_argument("--rank", default=10, type=int)
    parser.add_argument("--num_chunk", default=10, type=int)
    parser.add_argument("--n_sample_group", default=4, type=int)
    parser.add_argument("--grad_reduction", default="mean", type=str)

    # for MoEs
    parser.add_argument("--n_exp", default=3, type=int)
    parser.add_argument("--n_exp_shared", default=3, type=int)
    parser.add_argument("--exp_layer", default=2, type=int)
    parser.add_argument("--tower_layer", default=1, type=int)
    parser.add_argument("--exp_hidden", default=128, type=int)
    parser.add_argument("--exp_type", default="mlp", type=str)
    parser.add_argument("--gate_type", default="softmax", type=str)
    parser.add_argument("--output_type", default="moe", type=str)
    parser.add_argument("--topk", default=1, type=int)
    parser.add_argument("--init_ratio", default=0.1, type=float)

    # for fintune
    parser.add_argument("--pretrain_model_path", default=None, type=str)
    parser.add_argument("--pretrain_tuner_path", default=None, type=str)
    parser.add_argument('--tuner_type', default="lora", type=str)
    parser.add_argument('--target_modules', default=None, action=EvalAction)
    parser.add_argument('--enable_tuner', default=True, action=EvalAction)
    parser.add_argument('--num_tuner_online', default=1, type=int)
    parser.add_argument('--shuffle_tuners', default=False, type=EvalAction)
    ## for lora
    parser.add_argument('--r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=8, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    parser.add_argument('--init_lora_weights', default=True, action=EvalAction)
    parser.add_argument('--init_type', default="kaiming", type=str)
    parser.add_argument('--std_scale', default=0.01, type=float)
    parser.add_argument('--apply_bias', default=False, action=EvalAction)
    ## for fourier
    parser.add_argument('--n_frequency', default=100, type=int)
    parser.add_argument('--scale', default=0.1, type=float)
    parser.add_argument('--init_fourier_weights', default=False, action=EvalAction)
    parser.add_argument('--set_bias', default=False, action=EvalAction)
    parser.add_argument('--fc', default=1.0, type=float)
    parser.add_argument('--width', default=200., type=float)
    parser.add_argument("--share_entry", default=False, action=EvalAction)
    ## for adapter
    parser.add_argument('--nonlinearity', default="relu", type=str)
    ## for IA3
    parser.add_argument('--apply_attention', default=True, action=EvalAction)

    args = parser.parse_args()

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_num_threads(args.thread)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name in EXP_DICT:
        Exp = EXP_DICT[args.task_name]

    # setproctitle.setproctitle(args.task_name)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_lpl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.label_pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.auxi_lambda,
                args.rec_lambda,
                args.auxi_loss,
                args.module_first,
                args.des,
                ii
            )

            if not args.rerun and os.path.exists(os.path.join(args.results, setting)) and \
                os.path.exists(os.path.join(args.results, setting, "metrics.npy")):
                print(f">>>>>>>setting {setting} already run, skip")
                sys.exit()

            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_lpl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.label_pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.auxi_lambda,
            args.rec_lambda,
            args.auxi_loss,
            args.module_first,
            args.des,
            ii
        )

        if not args.rerun and os.path.exists(os.path.join(args.results, setting)) and \
            os.path.exists(os.path.join(args.results, setting, "metrics.npy")):
            print(f">>>>>>>setting {setting} already run, skip")
            sys.exit()

        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
