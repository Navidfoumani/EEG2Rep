import os
import numpy as np
import torch
import pickle
from scipy.signal import resample


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/srv/local/data/TUH/tuh3/tuh_eeg_abnormal/v3.0.0/edf/processed"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in train_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    train_data = np.concatenate(all_data, axis=0)
    train_label = np.concatenate(all_labels, axis=0)

    np.save('train_data', train_data, allow_pickle=True)
    np.save('train_label', train_label, allow_pickle=True)

    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in test_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    test_data = np.concatenate(all_data, axis=0)
    test_label = np.concatenate(all_labels, axis=0)

    np.save('test_data', test_data, allow_pickle=True)
    np.save('test_label', test_label, allow_pickle=True)

    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in val_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    val_data = np.concatenate(all_data, axis=0)
    val_label = np.concatenate(all_labels, axis=0)
    np.save('val_data', val_data, allow_pickle=True)
    np.save('val_label', val_label, allow_pickle=True)

    All_train_data = np.concatenate((train_data, val_data), axis=0)
    All_train_label = np.concatenate((train_label, val_label), axis=0)
    np.save('All_train_data', All_train_data, allow_pickle=True)
    np.save('All_train_label', All_train_label, allow_pickle=True)

    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_TUEV_dataloader(args):
    # set random seed
    seed = 4523
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = os.path.join(os.getcwd(), "datasets/TUEV/TUEV/edf")

    train_files = os.listdir(os.path.join(root, "processed_train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "processed_eval"))

    val_sub = np.random.choice(train_sub, size=int(
        len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), train_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_eval"), test_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(
            os.path.join(
                root, "processed_train"), val_files, args.sampling_rate
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in train_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    train_data = np.concatenate(all_data, axis=0)
    train_label = np.concatenate(all_labels, axis=0)

    np.save('train_data', train_data, allow_pickle=True)
    np.save('train_label', train_label, allow_pickle=True)

    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in test_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    test_data = np.concatenate(all_data, axis=0)
    test_label = np.concatenate(all_labels, axis=0)

    np.save('test_data', test_data, allow_pickle=True)
    np.save('test_label', test_label, allow_pickle=True)

    # Initialize empty lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate over the DataLoader
    for batch_data in val_loader:
        # Extract data and label tensors
        data_tensor, label_tensor = batch_data[0], batch_data[1]

        # Convert the tensors to numpy arrays and append to the lists
        numpy_data = data_tensor.numpy()
        numpy_labels = label_tensor.numpy()

        all_data.append(numpy_data)
        all_labels.append(numpy_labels)

    # Concatenate all batches along the first axis
    val_data = np.concatenate(all_data, axis=0)
    val_label = np.concatenate(all_labels, axis=0)
    np.save('val_data', val_data, allow_pickle=True)
    np.save('val_label', val_label, allow_pickle=True)

    All_train_data = np.concatenate((train_data, val_data), axis=0)
    All_train_label = np.concatenate((train_label, val_label), axis=0)
    np.save('All_train_data', All_train_data, allow_pickle=True)
    np.save('All_train_label', All_train_label, allow_pickle=True)

    print(len(train_files), len(val_files), len(test_files))
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y
    

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y