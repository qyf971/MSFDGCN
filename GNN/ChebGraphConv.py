import math
import torch
import torch.nn as nn
import torch.nn.init as init


def normalize_adjacency_matrix(adj_matrix, scale=True):
    """
    对邻接矩阵进行对称归一化和缩放归一化。

    :param adj_matrix: torch.Tensor，邻接矩阵，形状为 [N, N]
    :param scale: bool，是否进行缩放归一化
    :return: torch.Tensor，处理后的拉普拉斯矩阵，形状为 [N, N]
    """
    # 将邻接矩阵转换为浮点类型
    adj_matrix = adj_matrix.float()

    # 计算度矩阵 D
    degree_matrix = torch.diag(adj_matrix.sum(dim=1))

    # 计算 D^(-1/2)
    degree_matrix_inv_sqrt = torch.diag(torch.pow(degree_matrix.diagonal(), -0.5))

    # 处理度为 0 的情况（防止 NaN）
    degree_matrix_inv_sqrt[torch.isnan(degree_matrix_inv_sqrt)] = 0

    # 对称归一化拉普拉斯矩阵
    normalized_laplacian = torch.eye(adj_matrix.size(0)).to(adj_matrix.device) - torch.mm(
        torch.mm(degree_matrix_inv_sqrt, adj_matrix),
        degree_matrix_inv_sqrt
    )

    if scale:
        # 计算特征值的最大值
        lambda_max = torch.linalg.eigvalsh(normalized_laplacian).max().item()
        # 缩放归一化
        normalized_laplacian = 2 * normalized_laplacian / lambda_max - torch.eye(adj_matrix.size(0))

    return normalized_laplacian.to(adj_matrix.device)


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        :param x: [B, T, N, c_in]
        :return: [B, T, N, c_out]
        """
        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv