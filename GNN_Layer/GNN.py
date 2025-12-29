import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.data import Data, Batch

# 将邻接矩阵转为PyG要求格式
def convert_adj(adj: torch.Tensor):
    """
    将稠密邻接矩阵 adj 转换为 PyTorch Geometric 所需的 edge_index 和 edge_attr。

    Args:
        adj (Tensor): 形状为 [N, N] 的稠密邻接矩阵，可位于 GPU 或 CPU 上。

    Returns:
        edge_index (Tensor): 形状为 [2, E]，表示边的起点和终点索引。
        edge_attr (Tensor): 形状为 [E]，表示每条边的权重。
    """
    edge_index = (adj != 0).nonzero(as_tuple=False).t()  # [2, E]
    edge_attr = adj[edge_index[0], edge_index[1]]       # [E]
    return edge_index, edge_attr


# class GCN_Layer(nn.Module):
#     def __init__(self, device, adj, in_features, out_features):
#         super(GCN_Layer, self).__init__()
#         self.gcn = GCNConv(in_features, out_features)
#         self.device = device
#
#         # 邻接矩阵转换
#         edge_index, edge_attr = convert_adj(adj)
#         self.edge_index = edge_index  # [2, E]
#         self.edge_attr = edge_attr    # [E]
#         self.N = adj.shape[0]
#
#     def forward(self, x):
#         # x: [B, N, D]
#         B, N, D = x.shape
#         assert N == self.N, "Input node number must match adj node count"
#
#         # 展平输入
#         x = x.reshape(B * N, D)
#
#         # 构建 batch edge_index 和 edge_attr
#         batch_edge_index = []
#         batch_edge_attr = []
#         for i in range(B):
#             offset = i * N
#             batch_edge_index.append(self.edge_index + offset)
#             batch_edge_attr.append(self.edge_attr)
#
#         batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)   # [2, B*E]
#         batch_edge_attr = torch.cat(batch_edge_attr, dim=0).to(self.device)     # [B*E]
#
#         # 一次性执行 GCN
#         x = self.gcn(x, batch_edge_index, batch_edge_attr)
#
#         # reshape 回 [B, N, -1]
#         x = x.view(B, N, -1)
#         return x


# --- GCN 层实现 (针对批量共享图结构) ---
class GCN_Layer(nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        super(GCN_Layer, self).__init__()

        self.device = device

        # PyG GCN 模块
        self.gcn = GCNConv(in_features, out_features)

        # 邻接矩阵转换并注册为 Buffer
        edge_index, edge_attr = convert_adj(adj)
        self.N = adj.shape[0]

        # 使用 register_buffer 确保这些图结构参数随模型一起移动到 GPU/CPU
        self.register_buffer('edge_index', edge_index.to(device))  # [2, E]
        self.register_buffer('edge_attr', edge_attr.to(device))  # [E]

    def forward(self, x: torch.Tensor):
        """
        在批量数据上执行 GCN，其中每个样本共享相同的图结构。

        Args:
            x (torch.Tensor): 节点特征 [B, N, D]

        Returns:
            torch.Tensor: GCN 输出 [B, N, out_features]
        """
        B, N, D = x.shape
        assert N == self.N, "Input node number must match adj node count"

        # 1. 展平输入： [B, N, D] -> [B*N, D]
        x_flat = x.reshape(B * N, D)

        # 2. 构建批量 edge_index 和 edge_attr (高效张量操作)

        # A. 创建偏移量 [0, N, 2N, 3N, ..., (B-1)N]
        # 注意: 偏移量也需要在对应的设备上
        offsets = torch.arange(B, device=self.device) * N

        # B. 扩展 edge_index: [2, E] -> [2, B*E]
        # 使用 repeat 扩展边索引
        num_edges = self.edge_index.size(1)
        # edge_index 在第二维度重复 B 次
        batch_edge_index = self.edge_index.repeat(1, B)

        # C. 扩展并加偏移量
        # 偏移量需要重复 E 次 (每条边对应一个批次偏移量)
        # offsets.view(1, -1) 是 [1, B]
        # 扩展成 [1, B*E]
        offset_tensor = offsets.repeat_interleave(num_edges).view(1, -1)

        # 加到 edge_index 上
        batch_edge_index = batch_edge_index + offset_tensor

        # D. 扩展 edge_attr: [E] -> [B*E]
        batch_edge_attr = self.edge_attr.repeat(B)

        # 3. 执行 GCN
        x_out = self.gcn(x_flat, batch_edge_index, batch_edge_attr)

        # 4. reshape 回 [B, N, -1]
        x_out = x_out.view(B, N, -1)
        return x_out


class ChebConv_Layer(nn.Module):
    def __init__(self, device, adj, in_features, out_features, K):
        super().__init__()
        self.device = device
        self.N = adj.shape[0]
        self.chebconv = ChebConv(in_features, out_features, K).to(device)

        # 转换静态图结构
        edge_index, edge_attr = convert_adj(adj.cpu())
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def forward(self, x):
        """
        x: [B, N, D]
        return: [B, N, out_features]
        """
        B, N, D = x.shape
        assert N == self.N, "Input node count does not match adjacency matrix"

        # 展平输入为 [B*N, D]
        x = x.reshape(B * N, D)

        # 构建批量 edge_index 和 edge_attr
        batch_edge_index = []
        batch_edge_attr = []
        for i in range(B):
            offset = i * N
            batch_edge_index.append(self.edge_index + offset)
            batch_edge_attr.append(self.edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)  # [2, B*E]
        batch_edge_attr = torch.cat(batch_edge_attr, dim=0).to(self.device)    # [B*E]

        # 执行 ChebConv
        x = self.chebconv(x, batch_edge_index, batch_edge_attr)

        # reshape 回原始维度 [B, N, out_features]
        x = x.view(B, N, -1)
        return x


# class GAT_Layer(nn.Module):
#     def __init__(self, device, adj, in_features, out_features, edge_dim=1, heads=4):
#         super(GAT_Layer, self).__init__()
#         self.gat = GATConv(in_features, out_features, edge_dim=edge_dim, heads=heads, concat=False)
#         self.device = device
#
#         # 邻接矩阵转换，只做一次
#         edge_index, edge_attr = convert_adj(adj)
#         self.edge_index = edge_index   # [2, E]
#         self.edge_attr = edge_attr     # [E]
#         self.N = adj.shape[0]          # 节点数
#
#     def forward(self, x):
#         # x: [B, N, D]
#         B, N, D = x.shape
#         assert N == self.N, f"Input node count {N} != expected {self.N}"
#
#         # reshape 成 [B*N, D]
#         x = x.reshape(B * N, D)
#
#         # 批量构造 edge_index 和 edge_attr
#         batch_edge_index = []
#         batch_edge_attr = []
#         for i in range(B):
#             offset = i * N
#             batch_edge_index.append(self.edge_index + offset)
#             batch_edge_attr.append(self.edge_attr)
#
#         batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)  # [2, B*E]
#         batch_edge_attr = torch.cat(batch_edge_attr, dim=0).to(self.device)    # [B*E]
#         x = x.to(self.device)
#
#         # 执行 GATConv，获取 attention weights（可选）
#         x_out, attn_weights = self.gat(x, batch_edge_index, batch_edge_attr, return_attention_weights=True)
#
#         # reshape 回原始维度
#         x_out = x_out.view(B, N, -1)
#         # return x_out, attn_weights
#         return x_out


class GAT_Layer(nn.Module):
    def __init__(self, device, adj, in_features, out_features, edge_dim=1, heads=4):
        super(GAT_Layer, self).__init__()

        # heads=4, concat=False 意味着输出特征是 out_features * 1 (已聚合)
        self.gat = GATConv(in_features, out_features, edge_dim=edge_dim, heads=heads, concat=False)
        self.device = device

        # 邻接矩阵转换并注册为 Buffer
        edge_index, edge_attr = convert_adj(adj)
        self.N = adj.shape[0]
        self.E = edge_index.size(1)  # 边的数量

        # 使用 register_buffer 确保图结构参数随模型一起移动
        self.register_buffer('edge_index', edge_index.to(device))  # [2, E]
        self.register_buffer('edge_attr', edge_attr.to(device))  # [E]

    def forward(self, x: torch.Tensor):
        # x: [B, N, D]
        B, N, D = x.shape
        assert N == self.N, f"Input node count {N} != expected {self.N}"

        # 确保输入数据在正确设备上
        x = x.to(self.device)

        # 1. 展平输入： [B, N, D] -> [B*N, D]
        x_flat = x.reshape(B * N, D)

        # 2. 高效构建批量 edge_index 和 edge_attr

        # A. 创建偏移量 [0, N, 2N, ..., (B-1)N]
        offsets = torch.arange(B, device=self.device) * N

        # B. 扩展 edge_index: [2, E] -> [2, B*E]
        batch_edge_index = self.edge_index.repeat(1, B)

        # C. 扩展并加偏移量
        # 偏移量需要重复 E 次
        offset_tensor = offsets.repeat_interleave(self.E).view(1, -1)
        batch_edge_index = batch_edge_index + offset_tensor

        # D. 扩展 edge_attr: [E] -> [B*E]
        batch_edge_attr = self.edge_attr.repeat(B)

        # 3. 执行 GATConv
        # GATConv 返回 (节点输出, (edge_index, attention_weights))
        x_out_tuple = self.gat(x_flat, batch_edge_index, batch_edge_attr, return_attention_weights=True)

        x_out = x_out_tuple[0]
        # attn_weights = x_out_tuple[1] # 如果需要注意力权重，可以解包

        # 4. reshape 回原始维度
        x_out = x_out.view(B, N, -1)
        return x_out

