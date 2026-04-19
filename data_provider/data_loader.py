import os
import warnings
import h5py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

class Dataset_Loader(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='metr-la.csv',
                 scale=True, timeenc=0, freq='5min', percent=100):
        if size is None:
            self.seq_len = 12
            self.label_len = 3
            self.pred_len = 3
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.percent = percent
        self.scale = False
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # Read the raw CSV and create the split used by the current loader flag.
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # Keep the existing experimental cap to match the current training setup.
        max_data_len = 8000
        if len(df_raw) > max_data_len:
            df_raw = df_raw[:max_data_len]
        data_len = len(df_raw)
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        train_end = int(data_len * train_ratio)
        val_end = train_end + int(data_len * val_ratio)
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 5)
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            # Delegate calendar feature generation to the shared time-feature helper.
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
