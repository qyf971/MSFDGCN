import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from data.dataloader_recent import Dataloader_Recent
from _Support.Graph_Construction import calculate_adjacency_matrix
from utils.metrics import metric
from utils.tools import adjust_learning_rate, EarlyStopping, setup_seed
from model.MSFDGCN import Model


import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


class Exp_model:
    def __init__(self, model_name, model, epoch, learning_rate, target_state, batch_size, patience, num_workers, seq_len, pred_len):
        print(model)
        self.count_parameters(model)
        self.model_name = model_name
        self.model = model.cuda()
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.target_state = target_state
        self.batch_size = batch_size
        self.patience = patience
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.results_folder = f'results_dataset19to20_final_L{self.seq_len}_T{self.pred_len}/{self.model_name}_{self.target_state}_s{self.seq_len}_p{self.pred_len}/'

        os.makedirs(self.results_folder, exist_ok=True)

        self.dataloader = Dataloader_Recent(self.batch_size, self.num_workers, self.target_state, self.seq_len, self.pred_len)
        self.train_loader, self.val_loader, self.test_loader = self.dataloader.get_dataloader()

    def val(self, criterion):
        val_loss = []
        test_loss = []

        self.model.eval()
        for i, (features, target) in enumerate(self.val_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            val_loss.append(loss.item())
        val_loss = np.average(val_loss)

        for i, (features, target) in enumerate(self.test_loader):
            features = features.cuda()
            target = target.cuda()
            pred, true = self.model(features), target
            loss = criterion(pred, true)
            test_loss.append(loss.item())
        test_loss = np.average(test_loss)
        self.model.train()

        return val_loss, test_loss

    def train(self):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # 非常重要

        train_loss = []
        val_loss = []
        test_loss = []

        epoch_time = []

        self.model.train()
        time_start = time.time()

        # 初始化早停对象
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=os.path.join(self.results_folder, self.model_name + '.pth'))

        for epoch in range(self.epoch):
            adjust_learning_rate(optim, epoch + 1, self.epoch, self.learning_rate)

            epoch_train_loss = []
            epoch_start_time = time.time()

            for i, (features, target) in enumerate(self.train_loader):
                features = features.cuda()
                target = target.cuda()
                optim.zero_grad()
                pred, true = self.model(features), target
                loss = criterion(pred, true)
                epoch_train_loss.append(loss.item())
                loss.backward()
                optim.step()

            epoch_end_time = time.time()

            epoch_time.append(epoch_end_time - epoch_start_time)

            epoch_train_loss = np.average(epoch_train_loss)
            epoch_val_loss, epoch_test_loss = self.val(criterion)

            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            test_loss.append(epoch_test_loss)

            print(f"cost time:{epoch_end_time - epoch_start_time:.5f} train_loss:{epoch_train_loss:.5f} val_loss:{epoch_val_loss:.5f} test_loss:{epoch_test_loss:.5f}", end=" ")

            # 调用早停
            early_stopping(epoch_val_loss, self.model) # 非常重要

            if early_stopping.early_stop:
                print("Early stopping")
                break

        time_end = time.time()
        print("训练时间为{:.2f}秒，即{:.2f}分钟".format(time_end - time_start, (time_end - time_start) / 60))

        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        test_loss = np.array(test_loss)
        epoch_time = np.array(epoch_time)
        train_loss_df = pd.DataFrame({'epoch_time': epoch_time, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})
        train_loss_df.to_csv(os.path.join(self.results_folder, 'loss.csv'), index=True, index_label='epoch') # train_loss saving

        # torch.save(self.model.state_dict(), os.path.join(self.results_folder, self.model_name + '.pth')) # models saving

        # 加载早停保存的最佳模型
        best_model_path = os.path.join(self.results_folder, self.model_name + '.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def _evaluate(self, dataloader, loader, flag):
        predictions = []
        trues = []
        features_list = []

        total_inference_time = 0  # ⏱️ 记录总推理时间

        for i, (features, target) in enumerate(loader):
            features = features.cuda()
            features_np = features.cpu().numpy()
            features_list.append(features_np)

            with torch.no_grad():
                start_time = time.time()  # ⏱️ 开始计时
                pred = self.model(features)
                end_time = time.time()  # ⏱️ 结束计时
                total_inference_time += (end_time - start_time)

            true = target
            predictions.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        trues_inverse = dataloader.inverse_transform(trues)
        predictions_inverse = dataloader.inverse_transform(predictions)

        if flag == 'test':
            np.save(os.path.join(self.results_folder, f'{flag}_y_inverse.npy'), trues_inverse)
            np.save(os.path.join(self.results_folder, f'{flag}_predictions_inverse.npy'), predictions_inverse)

        metrics = metric(predictions_inverse, trues_inverse)

        # unpack five metrics
        mae, rmse, nmae, nrmse, r2 = metrics

        # printing metrics and training time
        print(
            f'{flag:>5}_MAE: {mae:<8.3f}, {flag:>5}_RMSE: {rmse:<8.3f}, {flag:>5}_nMAE: {nmae:<8.3f}, {flag:>5}_nRMSE: {nrmse:<8.3f}, {flag:>5}_R2: {r2:<8.3f}, '
            f'{flag:>5}_Inference_time: {total_inference_time:<8.3f}, {flag:>5}_Inference_time_per_batch: {total_inference_time / len(loader):<8.3f}'
        )

        # saving metrics
        if flag == 'test':
            metrics_with_time = list(metrics) + [total_inference_time]
            metrics_df = pd.DataFrame(
                [metrics_with_time],
                columns=[f'{flag}_MAE', f'{flag}_RMSE', f'{flag}_nMAE', f'{flag}_nRMSE', f'{flag}_R2', f'{flag}_Inference_time']
            )
            metrics_df.to_csv(os.path.join(self.results_folder, f'{flag}_metrics.csv'), index=False)

    def test(self):
        self.model.eval()

        self._evaluate(self.dataloader, self.train_loader, flag='train')
        self._evaluate(self.dataloader, self.val_loader, flag='val')
        self._evaluate(self.dataloader, self.test_loader, flag='test')

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# 模型参数
in_channels = 7
T_in = 96
T_out = 96
hidden_size = 64
num_layers = 2
layers = 2
target_state = 'Illinois_cities'
alpha = 0.2
n_heads = 4
kernel_size = 2
kernel_size_TCN = 4
dropout = 0.2
K=2
num_heads = 2
num_nodes = 8
num_channels = [64, 64, 64, 64]
num_of_blocks = 2
max_orders = 4


# 训练参数
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 128
epoch = 100
learning_rate = 0.0003
patience = 10
num_workers = 3

parser = argparse.ArgumentParser(description='model parameters')
parser.add_argument("--seq_len", type=int, default=T_in, help="input sequence length")
parser.add_argument("--pred_len", type=int, default=T_out, help="prediction sequence length")
parser.add_argument("--in_channels", type=int, default=in_channels, help="in_channels")
parser.add_argument("--d_model", type=int, default=hidden_size, help="d_model")
parser.add_argument("--out_channels", type=int, default=hidden_size, help="d_model")
parser.add_argument("--down_sampling_window", type=int, default=2, help="down_sampling_window")
parser.add_argument("--down_sampling_layers", type=int, default=2, help="down_sampling_layers")
parser.add_argument("--decomposition_method", type=str, default='moving_avg', help="decomposition_method")
parser.add_argument("--moving_avg", type=str, default=25, help="moving_avg")
parser.add_argument("--e_layers", type=int, default=2, help="e_layers")
parser.add_argument("--target_state", type=str, default=target_state, help="which state to forecast, options: [California_cities, Oregon_cities, Texas_cities]")
parser.add_argument("--output_attention", action='store_true', help='whether to output attention in encoder')
parser.add_argument("--use_norm", type=int, default=True, help='use norm and denorm')
parser.add_argument("--dropout", type=float, default=0.1, help='dropout')
parser.add_argument("--embed", type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument("--factor", type=int, default=1, help='attn factor')
parser.add_argument("--n_heads", type=int, default=4, help='num of heads')
parser.add_argument("--d_ff", type=int, default=2048, help='dimension of fcn')
parser.add_argument("--activation", type=str, default='gelu', help='activation')
parser.add_argument("--output_adj", type=bool, default=True, help='whether to output adj matrix')
parser.add_argument("--channel_independence", type=int, default=0, help='channel_independence')
args = parser.parse_args()



models = {
        'Model': Model(args, in_channels, hidden_size, 2, args.pred_len),

}

if __name__ == '__main__':
    setup_seed(2025)
    for model_name, model_instance in models.items():
        exp = Exp_model(model_name, model_instance, epoch, learning_rate, args.target_state, batch_size, patience, num_workers, args.seq_len, args.pred_len)
        print(f'model name: {model_name} seq_len: {args.seq_len} pred_len: {args.pred_len} target state: {args.target_state}')
        print("***************************************************************" + model_name + "训练开始！" + "***************************************************************")
        exp.train()
        print("***************************************************************" + model_name + "训练结束！" + "***************************************************************")
        print("***************************************************************" + model_name + "测试开始！" + "***************************************************************")
        exp.test()
        print("***************************************************************" + model_name + "测试结束！" + "***************************************************************" + "\n\n\n\n\n")
