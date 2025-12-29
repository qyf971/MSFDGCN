import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjacencyMatrixConstructor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, xt):
        # xt: [B, N, D]
        h = self.dense(xt)  # [B, N, out_dim]
        sim = torch.matmul(h, h.transpose(-1, -2))  # [B, N, N]
        sim_relu = F.relu(sim)
        attn = F.softmax(sim_relu, dim=-1)  # 每一行归一化
        adj = (attn + attn.transpose(-1, -2)) / 2  # 对称处理
        return adj  # [B, N, N]



class AdaptiveGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.adj_constructor = AdjacencyMatrixConstructor(input_dim, hidden_dim)

        self.gcn1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        # x: [B, N, D]
        B, N, _ = x.shape

        A = self.adj_constructor(x)  # [B, N, N]

        # GCN 层 1
        h = self.gcn1(x)     # [B, N, hidden_dim]
        h = torch.matmul(A, h)  # [B, N, hidden_dim]
        h = F.relu(h)

        # GCN 层 2
        h = self.gcn2(h)     # [B, N, hidden_dim]
        h = torch.matmul(A, h)  # [B, N, hidden_dim]
        h = F.relu(h)

        return h  # [B, N, hidden_dim]


# 测试代码
if __name__ == "__main__":
    N = 10
    input_dim = 16
    adj_hidden_dim = 8
    hidden_dim = 32
    output_dim = 2

    x = torch.randn(N, input_dim)
    model = AdaptiveGCN(input_dim, hidden_dim, adj_hidden_dim)
    out = model(x)
    print(out.shape)  # torch.Size([10, 2])
