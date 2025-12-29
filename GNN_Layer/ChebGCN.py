import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
import scipy.sparse as sp


# 将邻接矩阵转为PyG要求格式
def convert_adj_sparse(adj):
    adj_coo = sp.coo_matrix(adj.cpu().numpy())
    edge_attr = torch.FloatTensor(adj_coo.data).to(adj.device)
    edge_index = torch.LongTensor([adj_coo.row, adj_coo.col]).to(adj.device)
    return edge_index, edge_attr


class ChebGCN(nn.Module):
    def __init__(self, device, in_features, out_features, K=2):
        super(ChebGCN, self).__init__()
        self.conv = ChebConv(in_features, out_features, K=K).to(device)
        self.convert = convert_adj_sparse

    def forward(self, data, adj):
        edge_index, edge_attr = self.convert(adj)
        batch_size = data.size(0)
        num_nodes = adj.size(0)

        x = data.reshape(batch_size * num_nodes, -1)

        # 重复 edge_index
        edge_index = edge_index.repeat(1, batch_size)

        # 重复 edge_attr
        edge_attr = edge_attr.repeat(batch_size)

        x = self.conv(x, edge_index, edge_weight=edge_attr)

        x = x.reshape(batch_size, num_nodes, -1)
        return x


class ChebGCN_Layer(nn.Module):
    def __init__(self, device, in_features, out_features, K=3):
        super(ChebGCN_Layer, self).__init__()
        self.chebgcn = ChebGCN(device, in_features, out_features, K=K)

    def forward(self, data, adj):
        B, T, N, d = data.shape

        # 将数据展平为 (B * T, N, d)
        data_flattened = data.reshape(B * T, N, d)

        # 一次性传入 ChebGCN 层
        x = self.chebgcn(data_flattened, adj)

        # 将结果恢复为 (B, T, N, out_features)
        x = x.reshape(B, T, N, -1)

        return x
