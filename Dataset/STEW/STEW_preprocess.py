import os
import re
import pandas as pd
import numpy as np
import random
from scipy.signal import zpk2sos, sosfiltfilt, cheb2ord, iirdesign
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def STEW(window_size, step, current_path):
    # Build data
    Data = {}
    data_path = os.path.join(current_path, 'STEW/processed_data/STEW.npy')
    if os.path.exists(data_path):
        logger.info("Loading preprocessed STEW data ...")
        Data_npy = np.load(data_path, allow_pickle=True)
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        # Define the path to check
        raw_data_path = os.path.join(current_path, 'STEW/STEW Dataset')
        X_data, y_data, id_data = generate_data(raw_data_path)
        dataset_path = os.path.join(current_path, 'STEW', 'processed_data')
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        np.save(dataset_path + '/STEW_X.npy', X_data.astype(np.float32), allow_pickle=True)
        np.save(dataset_path + '/STEW_y.npy', y_data.astype(np.int64), allow_pickle=True)
        np.save(dataset_path + '/STEW_metadata.npy', {'subject_id': id_data}, allow_pickle=True)

        unique_groups = np.unique(id_data)
        n_test = int(len(unique_groups) * 0.25)
        for i in range(5):
            test_groups = np.random.choice(unique_groups, size=n_test, replace=False)
            # Get the indices where id_data matches the selected test groups
            test_indices = np.where(np.isin(id_data, test_groups))[0]
            logger.info("{} samples will be used for testing".format(len(test_indices)))
            # Write test indices to a text file
            with open(os.path.join(dataset_path, f"test_indices_fold_{i}.txt"), "w") as f:
                for idx in test_indices:
                    f.write(f"{idx}\n")
    return


def Windowed_majority_labeling(values, labels, ids, window_size, step):
    # Initialize empty lists to store windowed samples and labels
    windowed_samples = []
    window_labels = []
    window_ids = []

    for i in range(0, len(values) - window_size + 1, step):
        # Extract the windowed sample
        windowed_sample = values[i:i + window_size]

        # Assign the majority label to the window
        window_label = np.argmax(np.bincount(labels[i:i + window_size]))
        window_id = np.argmax(np.bincount(ids[i:i + window_size]))

        # Append the windowed sample and label to the lists
        windowed_samples.append(list(windowed_sample))
        window_labels.append(window_label)
        window_ids.append(window_id)

    # Convert the windowed samples and labels to numpy arrays
    windowed_samples = np.transpose(np.array(windowed_samples), (0, 2, 1))
    window_labels = np.array(window_labels)
    window_ids = np.array(window_ids)
    return windowed_samples, window_labels, window_ids


def generate_data(data_path):
    X_datas = np.empty((0, 14, 256))
    y_datas = np.empty(0)
    id_datas = np.array([], dtype=int)
    for file_name in os.listdir(data_path):
        if file_name.startswith('s'):
            # Determine the label based on the file name
            label = 1 if file_name.endswith('hi.txt') else 0
            # Construct the full path to the file
            file_path = os.path.join(data_path, file_name)
            # Read the CSV file into a dataframe
            values = pd.read_csv(file_path, delimiter='\s+', header=None)
            values = chebyBandpassFilter(values, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
            # values = scaler.fit_transform(values)
            X_data, y_data, id_data = Windowed_majority_labeling(values, np.full(len(values), int(label)),
                                                                 np.full(len(values), extract_sub_number(file_name)),
                                                                 window_size, step)
            X_datas = np.vstack((X_datas, X_data))
            y_datas = np.append(y_datas, y_data)
            id_datas = np.append(id_datas, id_data)

    return X_datas, y_datas, id_datas


def extract_sub_number(filename):
    # Use regular expression to find the number after "sub"
    match = re.search(r'sub(\d+)', filename)
    if match:
        # Extract the matched number and convert it to integer
        sub_number = int(match.group(1))
        return sub_number

def chebyBandpassFilter(data, cutoff, gstop=40, gpass=0.5, fs=128):
    """
    Design a filter with scipy functions avoiding unstable results (when using
    ab output and filtfilt(), lfilter()...).
    Cf. ()[]

    Parameters
    ----------
    data : instance of numpy.array | instance of pandas.core.DataFrame
        Data to be filtered. Each column will be filtered if data is a
        dataframe.
    cutoff : array-like of float
        Pass and stop frequencies in order:
            - the first element is the stop limit in the lower bound
            - the second element is the lower bound of the pass-band
            - the third element is the upper bound of the pass-band
            - the fourth element is the stop limit in the upper bound
        For instance, [0.9, 1, 45, 48] will create a band-pass filter between
        1 Hz and 45 Hz.
    gstop : int
        The minimum attenuation in the stopband (dB).
    gpass : int
        The maximum loss in the passband (dB).

    Returns:

    zpk :

    filteredData : instance of numpy.array | instance of pandas.core.DataFrame
        The filtered data.
    """

    wp = [cutoff[1]/(fs/2), cutoff[2]/(fs/2)]
    ws = [cutoff[0]/(fs/2), cutoff[3]/(fs/2)]

    z, p, k = iirdesign(wp = wp, ws= ws, gstop=gstop, gpass=gpass,
        ftype='cheby2', output='zpk')
    zpk = [z, p, k]
    sos = zpk2sos(z, p, k)

    order, Wn = cheb2ord(wp = wp, ws= ws, gstop=gstop, gpass=gpass, analog=False)

    print('Creating cheby filter of order %d...' % order)

    if (data.ndim == 2):
        print('Data contain multiple columns. Apply filter on each columns.')
        filteredData = np.zeros(data.shape)
        for electrode in range(data.shape[1]):
            # print 'Filtering electrode %s...' % electrode
            filteredData[:, electrode] = sosfiltfilt(sos, data[electrode].values)
    else:
        # Use sosfiltfilt instead of filtfilt fixed the artifacts at the beggining
        # of the signal
        filteredData = sosfiltfilt(sos, data)
    return filteredData


if __name__ == '__main__':
    window_size = 256
    step = 64
    # Get the current directory path
    current_path, _ = os.path.split(os.getcwd())
    STEW(window_size, step, current_path)

