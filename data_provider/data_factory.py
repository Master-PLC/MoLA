from torch.utils.data import DataLoader

from data_provider.data_loader import (Dataset_Custom, Dataset_Custom_PCA,
                                       Dataset_ETT_hour, Dataset_ETT_hour_PCA,
                                       Dataset_ETT_hour_Trend,
                                       Dataset_ETT_minute, Dataset_M4,
                                       Dataset_PEMS, Dataset_Solar,
                                       MSLSegLoader, PSMSegLoader,
                                       SMAPSegLoader, SMDSegLoader,
                                       SWATSegLoader, UEAloader)
from data_provider.uea import collate_fn

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_Trend': Dataset_ETT_hour_Trend,
    'ETTh1_PCA': Dataset_ETT_hour_PCA,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom_PCA': Dataset_Custom_PCA,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
    # if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.test_batch_size  # bsz for test
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.label_pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            add_noise=args.add_noise,
            noise_amp=args.noise_amp,
            noise_freq_percentage=args.noise_freq_percentage,
            noise_seed=args.noise_seed,
            noise_type=args.noise_type,
            data_percentage=args.data_percentage,
            rank_ratio=args.rank_ratio,
            pca_dim=args.pca_dim,
            reinit=args.reinit
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
