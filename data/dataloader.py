import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class Dataloader_Recent_Day_Week(object):
    def __init__(self, data_path, flag, batch_size, num_workers, target_state):
        self.data_path = data_path
        self.flag = flag
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_state = target_state
        self.read_data()

    def read_data(self):
        try:
            data = np.load(self.data_path)
            self.hour = torch.tensor(data['hour'], dtype=torch.float32).transpose(1, 3)  # [B, F, N, T]
            self.day = torch.tensor(data['day'], dtype=torch.float32).transpose(1, 3)  # [B, F, N, T]
            self.week = torch.tensor(data['week'], dtype=torch.float32).transpose(1, 3)  # [B, F, N, T]
            self.target_tensor = torch.tensor(data['target'], dtype=torch.float32).transpose(1, 2).squeeze()   # (B, N, T)

            # 检查数据形状一致性
            assert self.hour.shape[0] == self.day.shape[0] == self.week.shape[0] == self.target_tensor.shape[0], "数据形状不一致"

        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {self.data_path} 不存在")
        except Exception as e:
            raise RuntimeError(f"无法加载文件 {self.data_path}: {e}")

    def get_dataloader(self):
        if self.flag == 'train':
            shuffle_flag = True
            drop_last = True
        elif self.flag == 'val':
            shuffle_flag = False
            drop_last = False
        elif self.flag == 'test':
            shuffle_flag = False
            drop_last = False
        else:
            raise ValueError(f"未知的标志: {self.flag}")

        dataset = TensorDataset(self.hour, self.day, self.week, self.target_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last
        )
        print(f"{self.flag} 数据准备完成")
        return dataloader

    def inverse_transform(self, data):
        scaler_path = f'./GHI_norm_data/{self.target_state}/{self.target_state}_scaler.npy'
        scaler = np.load(scaler_path)
        mean, std = scaler[0].reshape(1, -1, 1), scaler[1].reshape(1, -1, 1)
        data_inverse = data * std + mean
        return data_inverse



class Dataloader_Recent(object):
    def __init__(self, data_path, flag, batch_size, num_workers, target_state):
        self.data_path = data_path
        self.flag = flag
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_state = target_state
        self.read_data()

    def read_data(self):
        try:
            data = np.load(self.data_path)
            self.hour = torch.tensor(data['hour'], dtype=torch.float32).transpose(1, 2)  # [B, N, T, F]
            self.target_tensor = torch.tensor(data['target'], dtype=torch.float32).transpose(1, 2).squeeze()  # [B, N, T]

            # 检查数据形状一致性
            assert self.hour.shape[0] == self.target_tensor.shape[0], "数据形状不一致"

        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {self.data_path} 不存在")
        except Exception as e:
            raise RuntimeError(f"无法加载文件 {self.data_path}: {e}")

    def get_dataloader(self):
        if self.flag == 'train':
            shuffle_flag = True
            drop_last = True
        elif self.flag == 'val':
            shuffle_flag = False
            drop_last = False
        elif self.flag == 'test':
            shuffle_flag = False
            drop_last = False
        else:
            raise ValueError(f"未知的标志: {self.flag}")

        dataset = TensorDataset(self.hour, self.target_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last
        )
        print(f"{self.flag} 数据准备完成")
        return dataloader

    def inverse_transform(self, data):
        scaler_path = f'./GHI_norm_data/{self.target_state}/{self.target_state}_scaler.npy'
        scaler = np.load(scaler_path)
        mean, std = scaler[0].reshape(1, -1, 1), scaler[1].reshape(1, -1, 1)
        data_inverse = data * std + mean
        return data_inverse


