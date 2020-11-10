from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

## 데이터를 pickle 형태로 저장
def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)

## pickle 형태의 데이터를 불러오기
def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

## 데이터를 불러올 때 index로 불러오기
def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size-1, len(dates)):
        cur_date = dates[idx].to_pydatetime()
        in_date = dates[idx - (window_size-1)].to_pydatetime()

        _in_period = (cur_date - in_date).days * 24 * 60 + (cur_date - in_date).seconds / 60

        if _in_period == (window_size-1):
            input_idx.append(list(range(idx - window_size+1, idx+1)))
    return input_idx


class TagDataset(Dataset):
    def __init__(self, file_path, input_dim, window_size=1, norm_path=None):

        self.input_dim = input_dim
        self.window_size = window_size

        ## 파일이 있는지 확인.
        assert os.path.isfile(file_path), "[{}] 파일이 없습니다.".format(file_path)

        ## 파일 불러오기
        df = load_pickle(file_path)

        ## 파일 유효성 확인(Nan 확인)
        assert not df.isnull().values.any(), 'Dataframe 내 NaN이 포함되어 있습니다.'

        ## Index 추출
        dates = list(df['date'])
        self.input_ids = make_data_idx(dates, window_size=window_size)

        ## 정규화 파일 확인.
        if norm_path:
            ## 파일이 있는지 확인.
            assert os.path.isfile(norm_path), "[{}] 파일이 없습니다.".format(norm_path)

            ## 파일 불러오기
            if '.csv' in norm_path:
                norm_df = pd.read_csv(norm_path)
            else:
                norm_df = load_pickle(norm_path)

            norm_df.columns = ['tag_name', 'mean', 'std']

            ## 파일 유효성 확인(Nan 확인)
            assert not norm_df.isnull().values.any(), 'Dataframe 내 NaN이 포함되어 있습니다.'


        ## 선택된 변수 Column Float으로 변경
        selected_column = []
        for var_name in df.columns.tolist():
            if 'var' in var_name:
                df[var_name] = pd.to_numeric(df[var_name], errors='coerce')
                selected_column.append(var_name)
                
                ## 정규화 과정
                if norm_path and (norm_df['tag_name'] == var_name).sum() > 0:
                    temp_mean = norm_df[norm_df['tag_name'] == var_name]['mean'].values[0]
                    temp_std = norm_df[norm_df['tag_name'] == var_name]['std'].values[0]
                    df[var_name] = (df[var_name] - temp_mean) / temp_std

        var_data = df[selected_column[:input_dim]].values

        ## pytorch  모델에 import할 수 있도록 이름 conversion
        def convert_name(name):
            label_names = ['normal', 'progsys', 'abnormal']
            for idx, temp_name in enumerate(label_names):
                if name == temp_name:
                    return idx
            return 0

        df['label'] = df['state'].apply(convert_name)
        label_data = df['label'].values
        
        ## Summary 용
        self.df = df.iloc[np.array(self.input_ids)[:, -1]]
        self.selected_column = selected_column[:input_dim]

        ## torch로 type을 변경.
        self.var_data = torch.tensor(var_data, dtype=torch.float)
        self.label_data = torch.tensor(label_data, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        temp_input_ids = self.input_ids[item]
        input_values = self.var_data[temp_input_ids]
        input_labels = self.label_data[temp_input_ids]

        return input_values, input_labels

## 결과를 저장하는 파일
class ResultWriter:
    def __init__(self, directory):
        """ Save training Summary to .csv
        input
            args: training args
            results: training results (dict)
                - results should contain a key name 'val_loss'
        """
        self.dir = directory
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, **results):
        now = datetime.now()
        date = "%s-%s %s:%s" % (now.month, now.day, now.hour, now.minute)
        self.writer.update({"date": date})
        self.writer.update(results)
        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.makedirs(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None




