import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime
# import mne
import logging

from Dataset import data_loader

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def Setup(args):
    """
        Input:
            args: arguments object from argparse
        Returns:
            config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = os.path.join(config['output_dir'], config['Training_mode'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # config['problem'] = config['data_dir'].split('/')[-1]
    output_dir = os.path.join(output_dir, config['data_dir'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])
    config['problem'] = os.path.basename(config['data_dir'])
    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def Initialization(config):
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device


def Data_Loader(config):
    if config['problem'] =='TUEV':
        Data = data_loader.tuev_loader(config)
    else:
        Data = data_loader.load(config)
    # Data = convert_frequency(config, Data)
    return Data


class dataset_class(Dataset):

    def __init__(self, data, label, patch_size):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)
        self.patch_size = patch_size
        # self.__padding__()

    def __padding__(self):
        origin_len = self.feature[0].shape[1]
        if origin_len % self.patch_size:
            padding_len = self.patch_size - (origin_len % self.patch_size)
            padding = np.zeros((len(self.feature), self.feature[0].shape[0], padding_len), dtype=np.float32)
            self.feature = np.concatenate([self.feature, padding], axis=-1)

    def __getitem__(self, ind):
        x = self.feature[ind]
        x = x.astype(np.float32)
        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)


def print_title(text):
    title = f"           {text}          "
    border = '*' * len(title)
    print(border)
    print(title)
    print(border)


def convert_frequency(config, Data):
    problem = config['data_dir'].split('/')[-1]
    Data['All_train_data'] = get_fft(Data['All_train_data'])
    Data['train_data'] = get_fft(Data['train_data'])
    Data['val_data'] = get_fft(Data['val_data'])
    Data['test_data'] = get_fft(Data['test_data'])
    Data['max_len'] = 10
    np.save(config['data_dir'] + "/" + problem + '_f', Data, allow_pickle=True)
    return Data


def get_fft(train_data):
    fs = 128  # Sampling rate (128 Hz)
    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    F_train = np.zeros((train_data.shape[0], train_data.shape[1], 10))
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            data = train_data[i][j]

            # Get real amplitudes of FFT (only in postive frequencies)
            fft_vals = np.absolute(np.fft.rfft(data))

            # Get frequencies for amplitudes in Hz
            fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

            # Take the mean of the fft amplitude for each EEG band
            k = 0
            for band in eeg_bands:
                freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                                   (fft_freq <= eeg_bands[band][1]))[0]
                F_train[i, j, k] = np.min(fft_vals[freq_ix])
                F_train[i, j, k+1] = np.max(fft_vals[freq_ix])
                # F_train[i, j, k + 2] = np.min(fft_vals[freq_ix])
                k = k + 1

    return F_train




