import argparse
import numpy as np
import os
import torch

from data_provider.data_factory import data_provider_at_extraction
from lib.models.revin import RevIN


class ExtractData:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:' + str(self.args.gpu)
        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)

    def _get_data(self, flag):
        data_set, data_loader = data_provider_at_extraction(self.args, flag)
        return data_set, data_loader

    def one_loop_forecasting(self, loader):
        x_original_all = []
        y_original_all = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x_original_all.append(batch_x)
            y_original_all.append(batch_y)

        x_original_arr = np.concatenate(x_original_all, axis=0)
        y_original_arr = np.concatenate(y_original_all, axis=0)

        data_dict = {}
        data_dict['x_original_arr'] = x_original_arr
        data_dict['y_original_arr'] = y_original_arr

        print(data_dict['x_original_arr'].shape, data_dict['y_original_arr'].shape)

        return data_dict


    def extract_data(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if args.classifiy_or_forecast == 'forecast':
            print('FORECASTING')

            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)

            # These have dimension [bs, ntime, nvars]
            print('-------------TRAIN-------------')
            train_data_dict = self.one_loop_forecasting(train_loader)
            save_files_forecasting(self.args.save_path, train_data_dict, 'train')

            print('-------------Val-------------')
            val_data_dict = self.one_loop_forecasting(vali_loader)
            save_files_forecasting(self.args.save_path, val_data_dict, 'val')

            print('-------------Test-------------')
            test_data_dict = self.one_loop_forecasting(test_loader)
            save_files_forecasting(self.args.save_path, test_data_dict, 'test')


def save_files_forecasting(path, data_dict, mode):
    np.save(path + mode + '_x_original.npy', data_dict['x_original_arr'])
    np.save(path + mode + '_y_original.npy', data_dict['y_original_arr'])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./Datasets/imputation_and_forecasting_data/exchange_rate/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Formers
    parser.add_argument('--enc_in', type=int, default=8,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # Save Location
    parser.add_argument('--save_path', type=str, default=False,
                        help='folder ending in / where we want to save the revin data to')

    parser.add_argument('--classifiy_or_forecast', type=str, required=False, help='compression_factor')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = ExtractData
    exp = Exp(args)  # set experiments
    exp.extract_data()