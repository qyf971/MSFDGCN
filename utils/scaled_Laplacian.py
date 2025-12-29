import numpy as np
from scipy.sparse.linalg import eigs
import torch

def scaled_Laplacian(W):  # 计算缩放的拉普拉斯矩阵
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]  #35

    cheb_polynomials = [np.identity(N), L_tilde.copy()]  

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])  

    return cheb_polynomials


def calculate_laplacian_with_self_loop(matrix):   # ChebGCN不使用 GCN中才使用 此处有误导之嫌，我非常讨厌
    matrix = matrix + torch.eye(matrix.size(0))  # 添加自环
    row_sum = matrix.sum(1)  # 计算节点的度
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten() # 得到每个节点的-0.5次幂
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0 # 避免除以 0 导致无穷大，做数值稳定处理
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)   # 构建对角矩阵D（-0.5）
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt) # 计算得到对称归一化的拉普拉斯矩阵
    )
    return normalized_laplacian
