import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import scipy.sparse as sp


# 将邻接矩阵转为PyG要求格式
def convert_adj_sparse(adj):
    adj_coo = sp.coo_matrix(adj.cpu().numpy())
    edge_attr = torch.FloatTensor(adj_coo.data).to(adj.device)
    edge_index = torch.LongTensor([adj_coo.row, adj_coo.col]).to(adj.device)
    return edge_index, edge_attr


class GCN(nn.Module):
    def __init__(self, device, in_features, out_features):
        super(GCN, self).__init__()
        self.gcn = GCNConv(in_features, out_features).to(device)
        self.convert = convert_adj_sparse

    def forward(self, data, adj):
        edge_index, _ = self.convert(adj)  # GCN不需要edge_attr
        batch_size = data.size(0)
        num_nodes = adj.size(0)

        x = data.reshape(batch_size * num_nodes, -1)

        # 重复 edge_index
        edge_index = edge_index.repeat(1, batch_size)

        x = self.gcn(x, edge_index)

        x = x.reshape(batch_size, num_nodes, -1)
        return x


class GCN_Layer(nn.Module):
    def __init__(self, device, in_features, out_features):
        super(GCN_Layer, self).__init__()
        self.gcn = GCN(device, in_features, out_features)

    def forward(self, data, adj):
        B, T, N, d = data.shape

        # 将数据展平为 (B * T, N, d)
        data_flattened = data.reshape(B * T, N, d)

        # 一次性传入 GCN 层
        x = self.gcn(data_flattened, adj)

        # 将结果恢复为 (B, T, N, out_features)
        x = x.reshape(B, T, N, -1)

        return x
