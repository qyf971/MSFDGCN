import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, d_model):
        super(DynamicGraphConv, self).__init__()
        self.q_linear = nn.Linear(in_dim, d_model)
        self.k_linear = nn.Linear(in_dim, d_model)
        self.out_proj_1 = nn.Linear(in_dim, out_dim)
        self.out_proj_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: [B, N, F]
        Q = self.q_linear(x)  # [B, N, d_model]
        K = self.k_linear(x)  # [B, N, d_model]

        # 动态邻接矩阵 A = softmax(QK^T / sqrt(d_model))
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)  # [B, N, N]
        A = F.softmax(attn_scores, dim=-1)

        # 第一层图卷积
        out = torch.matmul(A, x)  # [B, N, in_dim] Ax
        out = self.out_proj_1(out) # [B, N, out_dim] AxW1
        out = F.relu(out) # [B, N, out_dim] ReLU(AxW1)

        # 第二层图卷积
        out = torch.matmul(A, out) # [B, N, out_dim] AReLU(AxW1)
        out = self.out_proj_2(out) # [B, N, out_dim] AReLU(AxW1)W2
        # out = F.relu(out) # [B, N, out_dim] ReLU(AReLU(AxW1)W2)

        return out

class DynamicGCNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, d_model):
        super(DynamicGCNLayer, self).__init__()
        # 生成动态邻接矩阵的线性映射
        self.q_linear = nn.Linear(in_dim, d_model)
        self.k_linear = nn.Linear(in_dim, d_model)

        # GCN 两层权重
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: [B, N, F]，节点特征矩阵
        返回: [B, N, hidden_dim]
        """
        # 构建动态邻接矩阵
        Q = self.q_linear(x)  # [B, N, d_model]
        K = self.k_linear(x)  # [B, N, d_model]
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (Q.shape[-1] ** 0.5)  # [B, N, N]
        A_tilde = F.softmax(attn_scores, dim=-1)  # 每行归一化

        # 第一层 GCN + ReLU
        H1 = F.relu(torch.matmul(A_tilde, x) @ self.W1.weight.T + self.W1.bias)

        # 第二层 GCN + ReLU
        H2 = F.relu(torch.matmul(A_tilde, H1) @ self.W2.weight.T + self.W2.bias)

        return H2

