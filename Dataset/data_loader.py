import os
import numpy as np
import pandas as pd
import logging
from sklearn import model_selection
from scipy import signal

logger = logging.getLogger(__name__)


def load(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True)

        if np.any(Data_npy.item().get('val_data')):
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['val_data'] = Data_npy.item().get('val_data')
            Data['val_label'] = Data_npy.item().get('val_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[1]
            Data['All_train_data'] = Data_npy.item().get('All_train_data')
            Data['All_train_label'] = Data_npy.item().get('All_train_label')
            if config['Pre_Training'] == 'Cross-domain':
                Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data_npy)
                logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy.item().get('train_data'), Data_npy.item().get('train_label'), 0.1)
            Data['All_train_data'] = Data_npy.item().get('train_data')
            Data['All_train_label'] = Data_npy.item().get('train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[2]
    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def Cross_Domain_loader(domain_data):
    All_train_data = domain_data.item().get('All_train_data')
    All_train_label = domain_data.item().get('All_train_label')
    # Load DREAMER for Pre-Training
    DREAMER = np.load('Dataset/DREAMER/DREAMER.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, DREAMER.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, DREAMER.item().get('All_train_label')), axis=0)

    # Load Crowdsource for Pre-Training
    Crowdsource = np.load('Dataset/Crowdsource/Crowdsource.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, Crowdsource.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, Crowdsource.item().get('All_train_label')), axis=0)
    return All_train_data, All_train_label

def tuev_loader(config):
    Data = {}
    data_path = config['data_dir'] + '/' + config['problem']
    Data['train_data'] = np.load(data_path + '/' + 'train_data.npy', allow_pickle=True)
    Data['train_label'] = np.load(data_path + '/' + 'train_label.npy', allow_pickle=True)
    Data['val_data'] = np.load(data_path + '/' + 'val_data.npy', allow_pickle=True)
    Data['val_label'] = np.load(data_path + '/' + 'val_label.npy', allow_pickle=True)
    Data['All_train_data'] = np.load(data_path + '/' + 'All_train_data.npy', allow_pickle=True)
    Data['All_train_label'] =np.load(data_path + '/' + 'All_train_label.npy', allow_pickle=True)
    Data['test_data'] = np.load(data_path + '/' + 'test_data.npy', allow_pickle=True)
    Data['test_label'] = np.load(data_path + '/' + 'test_label.npy', allow_pickle=True)
    Data['max_len'] = Data['train_data'].shape[1]

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))
    return Data


def fine_tune_data(Data, label, samples_per_class):
    # Randomly select samples from each class
    selected_indices = []
    for class_label in np.unique(label):
        indices = np.where(label == class_label)[0]
        selected_indices.extend(np.random.choice(indices, size=samples_per_class))
    # Select the corresponding data and labels
    selected_data = Data[selected_indices]
    selected_labels = label[selected_indices]

    return selected_data, selected_labels


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


def load_Preprocessed_DD(file_path, norm=True):
    Data = {}
    all_data = pd.read_csv(file_path)
    # Clean noisy recording = 'X', 'EyesCLOSEDneutral', 'EyesOPENneutral', 'LateBoredomLap'
    all_data = clean_Preprocessed_DD(all_data)
    all_labels = all_data.label__desc  # separate label
    # ------------------------------------------------------------------------------
    # Remove extra information -----------------------------------------------------
    subject_index = all_data.subject_id__desc
    col_index = find_feat_col(all_data)  # Remove description
    value = all_data[col_index]  # Remove description
    # -----------------------------------------------------------------------------
    # Split data to Train, Validation and Test -----------------------------------

    subject_uniq = subject_index.unique()
    test_subjects = subject_uniq[
        np.random.choice(len(subject_uniq), 3, replace=False)]  # Randomly select 3 subjects for testing
    valid_subject = np.random.choice(subject_uniq, 1)  # Randomly select 1 subject for validation

    test_subject_index = np.isin(subject_index, test_subjects)
    valid_subject_index = np.array(subject_index == valid_subject[0])

    test_data = value[test_subject_index]
    test_label = all_labels[test_subject_index]

    valid_data = value[valid_subject_index]
    valid_label = all_labels[valid_subject_index]

    # The remaining subjects will be used for validation
    train_subject_index = ~(test_subject_index | valid_subject_index)
    train_data = value[train_subject_index]
    train_label = all_labels[train_subject_index]

    test_data = reshape_DD(test_data)
    valid_data = reshape_DD(valid_data)
    train_data = reshape_DD(train_data)
    '''
    test_data = preprocess_eeg_data(reshape_DD(test_data), high_freq=50, low_freq=0.1, fs=128)
    valid_data = preprocess_eeg_data(reshape_DD(valid_data), high_freq=50, low_freq=0.1, fs=128)
    train_data = preprocess_eeg_data(reshape_DD(train_data), high_freq=50, low_freq=0.1, fs=128)
    '''


    logger.info("{} samples will be used for training".format(len(train_label)))
    logger.info("{} samples will be used for validation".format(len(valid_label)))
    logger.info("{} samples will be used for testing".format(len(test_label)))

    Data['max_len'] = train_data.shape[-1]
    Data['All_train_data'] = np.concatenate([train_data, valid_data])
    Data['All_train_label'] = np.concatenate([train_label.to_numpy(), valid_label.to_numpy()])
    Data['train_data'] = train_data
    Data['train_label'] = train_label.to_numpy()
    Data['val_data'] = valid_data
    Data['val_label'] = valid_label.to_numpy()
    Data['test_data'] = test_data
    Data['test_label'] = test_label.to_numpy()
    return Data


def load_Crowdsource(file_path):
    Data = {}
    all_data = pd.read_csv(file_path)
    label_type = all_data.labels_i.unique()
    all_data.labels_i = all_data.labels_i.replace(label_type[1], 1)
    all_data.labels_i = all_data.labels_i.replace(label_type[0], 0)

    all_labels = all_data.labels_i
    # Remove extra information -----------------------------------------------------
    subject_index = all_data.subject_id__desc
    col_index = find_feat_col(all_data)  # Remove description
    value = all_data[col_index]  # Remove description

    subject_uniq = subject_index.unique()
    test_subjects = subject_uniq[
        np.random.choice(len(subject_uniq), 3, replace=False)]  # Randomly select 3 subjects for testing

    valid_subject = test_subjects[-1]
    test_subjects = test_subjects[0:-1]

    test_subject_index = np.isin(subject_index, test_subjects)
    valid_subject_index = np.array(subject_index == valid_subject)

    test_data = value[test_subject_index]
    test_label = all_labels[test_subject_index]

    valid_data = value[valid_subject_index]
    valid_label = all_labels[valid_subject_index]

    # The remaining subjects will be used for validation
    train_subject_index = ~(test_subject_index | valid_subject_index)
    train_data = value[train_subject_index]
    train_label = all_labels[train_subject_index]

    test_data = reshape_DD(test_data)
    valid_data = reshape_DD(valid_data)
    train_data = reshape_DD(train_data)

    logger.info("{} samples will be used for training".format(len(train_label)))
    logger.info("{} samples will be used for validation".format(len(valid_label)))
    logger.info("{} samples will be used for testing".format(len(test_label)))

    Data['max_len'] = train_data.shape[-1]
    Data['All_train_data'] = np.concatenate([train_data, valid_data])
    Data['All_train_label'] = np.concatenate([train_label.to_numpy(), valid_label.to_numpy()])
    Data['train_data'] = train_data
    Data['train_label'] = train_label.to_numpy()
    Data['val_data'] = valid_data
    Data['val_label'] = valid_label.to_numpy()
    Data['test_data'] = test_data
    Data['test_label'] = test_label.to_numpy()

    return Data


def find_feat_col(data):
    column_list = data.columns.values.tolist()
    filter_col = filter(lambda a: 'feat' in a, column_list)
    index = list(filter_col)
    return index


def clean_Preprocessed_DD(all_data):
    # Drop other class data ------------------------------------------------------------------------------
    noise_class = ['x', 'EyesCLOSEDneutral', 'EyesOPENneutral', 'LateBoredomLap']
    all_labels = all_data.label__desc  # spliting the labels
    all_data = all_data.drop(np.squeeze(np.where(np.isin(all_labels, noise_class))))  # dropping the other classes

    # Added by Saad Irtza ---------------------------------------------------------
    cq_boolean = np.array(all_data['minimum_cq__desc'] > 3)
    all_data = all_data[cq_boolean]
    # ----------------------------------------------------------------------------
    Distraction = all_data.label__desc.unique()
    Distraction = np.setdiff1d(Distraction, 'Driving')
    Distraction = np.setdiff1d(Distraction, 'BoredomLap')
    all_data.label__desc = all_data.label__desc.replace(Distraction, 1)
    all_data.label__desc = all_data.label__desc.replace('Driving', 0)
    all_data.label__desc = all_data.label__desc.replace('BoredomLap', 0)
    return all_data


def reshape_DD(data):
    data = data.values.reshape(data.shape[0], 14, 256)  # Reshape to 2D
    return data


def preprocess_eeg_data(eeg_data, high_freq, low_freq, fs):
    """
    Preprocess EEG data by applying high-pass and low-pass filters and normalization.

    Args:
        eeg_data (np.ndarray): EEG data tensor of shape (num_samples, num_channels, length).
        high_freq (float): High-pass filter cutoff frequency in Hz.
        low_freq (float): Low-pass filter cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.

    Returns:
        preprocessed_data (np.ndarray): Preprocessed EEG data.
    """
    num_samples, num_channels, length = eeg_data.shape
    preprocessed_data = np.zeros_like(eeg_data)

    for channel in range(num_channels):
        # High-pass filter
        b_high, a_high = signal.butter(4, high_freq, btype='high', fs=fs)
        for sample in range(num_samples):
            eeg_data[sample, channel, :] = signal.lfilter(b_high, a_high, eeg_data[sample, channel, :])

        # Low-pass filter
        b_low, a_low = signal.butter(4, low_freq, btype='low', fs=fs)
        for sample in range(num_samples):
            eeg_data[sample, channel, :] = signal.lfilter(b_low, a_low, eeg_data[sample, channel, :])

        # Normalize using min-max scaling
        min_val = np.min(eeg_data[:, channel, :])
        max_val = np.max(eeg_data[:, channel, :])
        preprocessed_data[:, channel, :] = (eeg_data[:, channel, :] - min_val) / (max_val - min_val)

    return preprocessed_data
