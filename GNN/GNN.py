import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from torch_geometric.data import Data, Batch

from GNN.GCN import GCN_Layer

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


class GCN_Layer(torch.nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        super(GCN_Layer, self).__init__()
        self.gcn = GCNConv(in_features, out_features)
        self.device = device

        # 邻接矩阵转换
        edge_index, edge_attr = convert_adj(adj)
        self.edge_index = edge_index  # [2, E]
        self.edge_attr = edge_attr    # [E]
        self.N = adj.shape[0]

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        assert N == self.N, "Input node number must match adj node count"

        # 展平输入
        x = x.reshape(B * N, D)

        # 构建 batch edge_index 和 edge_attr
        batch_edge_index = []
        batch_edge_attr = []
        for i in range(B):
            offset = i * N
            batch_edge_index.append(self.edge_index + offset)
            batch_edge_attr.append(self.edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)   # [2, B*E]
        batch_edge_attr = torch.cat(batch_edge_attr, dim=0).to(self.device)     # [B*E]

        # 一次性执行 GCN
        x = self.gcn(x, batch_edge_index, batch_edge_attr)

        # reshape 回 [B, N, -1]
        x = x.view(B, N, -1)
        return x



class ChebConv_Layer(torch.nn.Module):
    def __init__(self, device, in_features, out_features, K):
        super(ChebConv_Layer, self).__init__()
        self.chebconv = ChebConv(in_features, out_features, K).to(device)
        self.convert = convert_adj
        self.device = device

    def forward(self, data, adj):
        adj = adj.cpu()
        data = data.cpu()
        data_adj, data_edge_features = self.convert(adj)
        data_list = [Data(x=x_, edge_index=data_adj, edge_attr=data_edge_features) for x_ in data]
        batch = Batch.from_data_list(data_list).to(self.device)
        x = self.chebconv(batch.x, batch.edge_index, batch.edge_attr)
        x = x.view(len(data), len(adj), -1)
        return x

class GAT_Layer(torch.nn.Module):
    def __init__(self, device, adj, in_features, out_features, edge_dim=1):
        super(GAT_Layer, self).__init__()
        self.gat = GATConv(in_features, out_features, edge_dim=edge_dim)
        self.device = device

        # 邻接矩阵转换，只做一次
        edge_index, edge_attr = convert_adj(adj)
        self.edge_index = edge_index   # [2, E]
        self.edge_attr = edge_attr     # [E]
        self.N = adj.shape[0]          # 节点数

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        assert N == self.N, f"Input node count {N} != expected {self.N}"

        # reshape 成 [B*N, D]
        x = x.reshape(B * N, D)

        # 批量构造 edge_index 和 edge_attr
        batch_edge_index = []
        batch_edge_attr = []
        for i in range(B):
            offset = i * N
            batch_edge_index.append(self.edge_index + offset)
            batch_edge_attr.append(self.edge_attr)

        batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.device)  # [2, B*E]
        batch_edge_attr = torch.cat(batch_edge_attr, dim=0).to(self.device)    # [B*E]
        x = x.to(self.device)

        # 执行 GATConv，获取 attention weights（可选）
        x_out, attn_weights = self.gat(x, batch_edge_index, batch_edge_attr, return_attention_weights=True)

        # reshape 回原始维度
        x_out = x_out.view(B, N, -1)
        return x_out, attn_weights

