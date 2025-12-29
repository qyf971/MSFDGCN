
import numpy as np
from sklearn.metrics import r2_score


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


# def R2(pred, true):
#     return r2_score(pred, true)
def nRMSE(y_pred, y_true, normalization='mean'):
    """
    计算每个站点的归一化 RMSE（nRMSE），并返回所有站点 nRMSE 的平均值。

    参数:
        y_pred: ndarray, shape [samples, N, T_out]，预测值
        y_true: ndarray, shape [samples, N, T_out]，真实值
        normalization: str, 'mean' 或 'range'

    返回:
        float: 所有站点 nRMSE 的平均值
    """
    mse = np.mean((y_pred - y_true) ** 2, axis=(0, 2))  # [N]
    rmse = np.sqrt(mse)  # [N]

    if normalization == 'mean':
        normalizer = np.mean(y_true, axis=(0, 2))  # [N]
    elif normalization == 'range':
        normalizer = np.max(y_true, axis=(0, 2)) - np.min(y_true, axis=(0, 2))  # [N]
    else:
        normalizer = np.ones_like(rmse)  # 默认不归一化

    normalizer = np.where(normalizer == 0, 1e-8, normalizer)  # 避免除以0
    nrmse = rmse / normalizer

    return np.mean(nrmse)

def nMAE(y_pred, y_true, normalization='mean'):
    """
    计算每个站点的归一化 MAE（nMAE），并返回所有站点 nMAE 的平均值。

    参数:
        y_pred: ndarray, shape [samples, N, T_out]，预测值
        y_true: ndarray, shape [samples, N, T_out]，真实值
        normalization: str, 'mean' 或 'range'

    返回:
        float: 所有站点 nMAE 的平均值
    """
    mae = np.mean(np.abs(y_pred - y_true), axis=(0, 2))  # [N]

    if normalization == 'mean':
        normalizer = np.mean(y_true, axis=(0, 2))  # [N]
    elif normalization == 'range':
        normalizer = np.max(y_true, axis=(0, 2)) - np.min(y_true, axis=(0, 2))  # [N]
    else:
        normalizer = np.ones_like(mae)  # 默认不归一化

    normalizer = np.where(normalizer == 0, 1e-8, normalizer)  # 避免除以0
    nmae = mae / normalizer

    return np.mean(nmae)

def R2(pred, true):
    """
    计算 R²（决定系数）。

    :param pred: 预测值，形状为 [samples, N, T_out]
    :param true: 真实值，形状为 [samples, N, T_out]
    :return: R² 值
    """
    # 将数据重塑为二维数组以适应 r2_score 函数
    samples, N, T_out = pred.shape
    pred_reshaped = pred.reshape(samples * N, T_out)
    true_reshaped = true.reshape(samples * N, T_out)
    r2 = r2_score(true_reshaped, pred_reshaped)
    return r2

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    nmae = nMAE(pred, true)
    nrmse = nRMSE(pred, true)
    r2 = R2(pred, true)
    return mae, rmse, nmae * 100, nrmse * 100, r2