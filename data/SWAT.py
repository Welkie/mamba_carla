import os
import pandas
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.mypath import MyPath
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SWAT(Dataset):
    """`SMD <https://www>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a ts
            and returns a transformed version.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('swat'), train=True, transform=None, sanomaly= None, mean_data=None, std_data=None):

        if 'SWAT_DATASET_PATH' in os.environ:
            root = os.environ['SWAT_DATASET_PATH']
        
        super(SWAT, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        labels = []
        self.wsz, self.stride = 512, 10

        if fname.lower() == 'swat':
            if self.train:
                file_name = 'normal.csv'
            else:
                file_name = 'attack.csv'
        else:
            if self.train:
                file_name = fname + '_train.csv'
            else:
                file_name = fname + '_test.csv'

        file_path = os.path.join(self.root, file_name)
        temp = pd.read_csv(file_path)
        
        # Strip whitespace from column names
        temp = temp.rename(columns=lambda x: x.strip())
        
        # Find the label column
        label_col = None
        possible_label_cols = ['attack', 'Attack', 'Normal/Attack', 'label', 'class']
        for col in possible_label_cols:
            if col in temp.columns:
                label_col = col
                break
        
        if label_col is None:
            raise KeyError(f"Could not find a label column in {file_path}. Available columns: {list(temp.columns)}")
            
        # Extract and convert labels
        raw_labels = temp[label_col].values
        if raw_labels.dtype == object or isinstance(raw_labels[0], str):
            # Map 'Normal' to 0, others to 1
            labels = (raw_labels != 'Normal').astype(int)
        else:
            labels = np.asarray(raw_labels)
            
        # Drop timestamp + label columns dynamically
        feature_df = temp.drop(columns=[label_col], errors='ignore')

        # Drop timestamp-like column
        for col in feature_df.columns:
            if 'time' in col.lower():
                feature_df = feature_df.drop(columns=[col])
                break

        temp = feature_df.values

        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean, self.std = mean_data, std_data

        if self.train:
            min_column = np.amin(temp, axis=0)
            max_column = np.amax(temp, axis=0)
            self.mean, self.std = min_column, max_column 
        else:
            self.mean, self.std = mean_data, std_data
            range_val = (std_data - mean_data) + 1e-20
            temp = (temp - mean_data) / range_val

        self.targets = labels
        self.data = np.asarray(temp)

        # Pre-calculate window start indices for lazy loading
        self.sliding_window_start_indices = np.arange(0, self.data.shape[0] - self.wsz + 1, self.stride)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        # Lazy loading of windows
        idx = self.sliding_window_start_indices[index]
        window = self.data[idx : idx + self.wsz]
        window_targets = self.targets[idx : idx + self.wsz]
        
        target = 1 if np.any(window_targets > 0) else 0
        class_name = self.classes[target]
        
        ts_org = torch.from_numpy(window).float().to(device)
        target = torch.tensor(target, dtype=torch.long).to(device)

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {'ts_org': ts_org, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.sliding_window_start_indices)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")