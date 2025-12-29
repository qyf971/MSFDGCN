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
        A = self.adj_constructor(x)  # [B, N, N]

        # GCN 层 1
        out = torch.matmul(A, x)
        out = self.gcn1(out)
        out = F.relu(out)

        # GCN 层 2
        out = torch.matmul(A, out)
        out = self.gcn2(out)
        out = F.relu(out)

        return out  # [B, N, hidden_dim] [B, N, N]

# 测试代码
if __name__ == "__main__":
    batch = 128
    N = 10
    input_dim = 16
    adj_hidden_dim = 8
    hidden_dim = 32
    output_dim = 2

    x = torch.randn(batch, N, input_dim)
    model = AdaptiveGCN(input_dim, hidden_dim)
    out = model(x)
    print(out.shape)
