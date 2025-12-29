import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Dataloader_Recent(object):
    def __init__(self, batch_size, num_workers, target_state, seq_len, pred_len):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_state = target_state
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.read_data()

    def read_data(self):
        data = np.load(f'./GHI_train_val_test_data_recent/{self.target_state}_h{self.seq_len}_p{self.pred_len}.npz')
        self.train_X = torch.tensor(data['X_train'], dtype=torch.float32).transpose(1, 2)
        self.train_y = torch.tensor(data['Y_train'], dtype=torch.float32).transpose(1, 2)
        self.val_X = torch.tensor(data['X_val'], dtype=torch.float32).transpose(1, 2)
        self.val_y = torch.tensor(data['Y_val'], dtype=torch.float32).transpose(1, 2)
        self.test_X = torch.tensor(data['X_test'], dtype=torch.float32).transpose(1, 2)
        self.test_y = torch.tensor(data['Y_test'], dtype=torch.float32).transpose(1, 2)

    def get_dataloader(self):
        train_dataset = TensorDataset(self.train_X, self.train_y)
        val_dataset = TensorDataset(self.val_X, self.val_y)
        test_dataset = TensorDataset(self.test_X, self.test_y)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        print(f"训练集准备完成 {len(train_dataset)}")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        print(f"验证集准备完成 {len(val_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        print(f"测试集准备完成 {len(test_dataset)}")
        return train_dataloader, val_dataloader, test_dataloader

    def inverse_transform(self, data):
        scaler_path = f'./GHI_norm_data/{self.target_state}/{self.target_state}_scaler.npy'
        scaler = np.load(scaler_path)
        mean, std = scaler[0].reshape(1, -1, 1), scaler[1].reshape(1, -1, 1)
        data_inverse = data * std + mean
        return data_inverse