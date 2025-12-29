from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATConv,  BatchNorm # noqa
import torch.nn as nn
from torch_geometric.nn import GINConv,  BatchNorm
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv, SAGEConv,GraphConv,ChebConv
from torch_geometric.utils import *
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch import Tensor
from torch.nn import Parameter
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian

class kanChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = torch.nn.ModuleList([
            KANLinear(in_channels, out_channels, 5, 3, 0.1, 1.0, 1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1,1]) for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        # for lin in self.lins:
        #     lin.reset_parameters()
        zeros(self.bias)


    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    # def forward(
    #     self,
    #     x: Tensor,
    #     edge_index: Tensor,
    #     edge_weight: OptTensor = None,
    #     batch: OptTensor = None,
    #     lambda_max: OptTensor = None,
    # ) -> Tensor:
    #
    #     edge_index, norm = self.__norm__(
    #         edge_index,
    #         x.size(self.node_dim),
    #         edge_weight,
    #         self.normalization,
    #         lambda_max,
    #         dtype=x.dtype,
    #         batch=batch,
    #     )
    #
    #     Tx_0 = x
    #     Tx_1 = x  # Dummy.
    #     out = self.lins[0](Tx_0)
    #
    #     # propagate_type: (x: Tensor, norm: Tensor)
    #     if len(self.lins) > 1:
    #         Tx_1 = self.propagate(edge_index, x=x, norm=norm)
    #         out = out + self.lins[1](Tx_1)
    #
    #     for lin in self.lins[2:]:
    #         Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
    #         Tx_2 = 2. * Tx_2 - Tx_0
    #         out = out + lin.forward(Tx_2)
    #         Tx_0, Tx_1 = Tx_1, Tx_2
    #
    #     if self.bias is not None:
    #         out = out + self.bias
    #     return out

    def forward(
            self,
            x: torch.Tensor,  # [B, N, F]
            edge_index: torch.Tensor,  # [2, E], 图的边索引
            edge_weight: torch.Tensor = None,  # [E], 边权
            lambda_max: torch.Tensor = None,
    ) -> torch.Tensor:
        B, N, F = x.size()

        # 1. 对单图 edge_index 和 edge_weight 做归一化
        edge_index_norm, edge_weight_norm = self.__norm__(
            edge_index,
            num_nodes=N,
            edge_weight=edge_weight,
            normalization=self.normalization,
            lambda_max=lambda_max,
            dtype=x.dtype
        )

        # 2. 生成 batch edge_index 和 edge_weight
        device = x.device
        edge_index_batch = []
        for b in range(B):
            offset = b * N
            edge_index_batch.append(edge_index_norm + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # [2, B*E]
        edge_weight_batch = edge_weight_norm.repeat(B)  # [B*E]

        # 3. 将 x 展平为 [B*N, F]
        x_flat = x.reshape(B * N, F)

        # 4. Chebyshev 多项式初始化
        Tx_0 = x_flat
        Tx_1 = x_flat
        out = self.lins[0](Tx_0)

        # 5. K ≥ 2 的情况，逐阶传播
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index_batch, x=x_flat, norm=edge_weight_batch)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index_batch, x=Tx_1, norm=edge_weight_batch)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        # 6. 加偏置
        if self.bias is not None:
            out = out + self.bias

        # 7. reshape 回 [B, N, out_channels]
        out = out.view(B, N, self.out_channels)
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

class kanGCNNet(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.conv1 = kanChebConv(graph.num_features, 512, K=1)
        self.conv2 = kanChebConv(512, 256, K=2)
        self.conv3 = kanChebConv(256, 128, K=4)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.fc = KANLinear(128, 6)

    def forward(self,graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr  # the Forward path of model
        x = F.relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.ln2(self.conv2(x, edge_index, edge_weight)))
        x = self.ln3(self.conv3(x, edge_index, edge_weight))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GCNNet768(torch.nn.Module):
    def __init__(self,graph):
        super(GCNNet768, self).__init__()
        self.conv1 = ChebConv(graph.num_features, 512, K=1)
        self.conv2 = ChebConv(512, 256, K=2)
        self.conv3 = ChebConv(256, 128, K=3)
        self.fc = torch.nn.Linear(128, 6)
        # self.conv1 = ChebConv(graph.num_features, 384, K=1)
        self.bn1 = BatchNorm(512)
        # self.conv2 = ChebConv(384, 192, K=2)
        self.bn2 = BatchNorm(256)
        # self.conv3 = ChebConv(192, 96, K=3)
        self.bn3 = BatchNorm(128)
        # self.fc = torch.nn.Linear(96, 6)

    def forward(self,graph):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr  # the Forward path of model
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = self.bn3(self.conv3(x, edge_index, edge_weight))
        x = self.fc(x)
        # x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.relu(self.conv2(x, edge_index, edge_weight))
        # x = self.conv3(x, edge_index, edge_weight)
        # x = self.fc(x)
        # x = torch.dropout(input=x,p=0.3,train=False)
        return F.log_softmax(x, dim=1)


class GraphKANConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphKANConv, self).__init__(aggr='add')  # 聚合方式：求和
        self.kan_linear = KANLinear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 如果edge_weight为空，初始化为1
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        else:
            # 添加自环时也要添加对应的权重
            edge_weight = torch.cat([edge_weight, torch.ones(x.size(0), device=x.device)])

        # 计算归一化权重
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return self.kan_linear(aggr_out)

class GraphKAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphKAN, self).__init__()
        self.conv1 = GraphKANConv(in_channels, hidden_channels)
        self.conv2 = GraphKANConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x