import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def normalize_adj(adj_matrix):
    """Row-normalize sparse matrix"""
    row_sum = np.array(adj_matrix.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def plot_loss(train_loss, val_loss, test_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Test Loss')
    plt.show()


# def adjust_learning_rate(optimizer, epoch, total_epoch, start_lr):
#     if epoch <= 5:
#         lr = start_lr
#     else:
#         lr = start_lr * (0.9 ** (epoch - 5))
#
#     print('Epoch [{:<3}/{:<3}] Learning Rate: {:.2e}'.format(epoch, total_epoch, lr), end=" ")
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, total_epoch, start_lr):
    hold_epochs = 5  # 保持不变的 Epoch 数 (1 到 5)
    decay_interval = 5  # 衰减发生的间隔 Epoch 数
    decay_factor = 0.5  # 衰减因子

    # 1. Hold 期 (Epoch <= 5)
    if epoch <= hold_epochs:
        lr = start_lr

    # 2. 阶梯衰减期 (Epoch >= 6)
    else:
        # 计算已发生的衰减次数 (k)
        # 目标:
        # Epoch 6-10: k=1 (衰减 1 次， factor 0.8^1)
        # Epoch 11-15: k=2 (衰减 2 次， factor 0.8^2)

        # 衰减周期内的 Epoch 编号 (从 1 开始)
        epoch_in_decay_period = epoch - hold_epochs

        # 使用 ceil 逻辑计算衰减步数 k
        # 在 Python 整数运算中，(A + B - 1) // B 等同于 ceil(A / B)
        decay_steps = (epoch_in_decay_period + decay_interval - 1) // decay_interval

        lr = start_lr * (decay_factor ** decay_steps)

    print('Epoch [{:<3}/{:<3}] Learning Rate: {:.2e}'.format(epoch, total_epoch, lr), end=" ")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_RLMC(optimizer, epoch, start_lr):
    lr = start_lr * (0.8 ** epoch)

    print('Epoch [{:<3}/{:<3}], Learning Rate: {:.2e}'.format(epoch, 100, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# 定义早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 允许验证集性能下降的次数
            verbose (bool): 是否打印早停信息
            delta (float): 最小性能提升的阈值
            path (str): 模型保存的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss