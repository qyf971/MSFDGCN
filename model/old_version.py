import torch
import torch.nn as nn
import torch.nn.functional as F
from _Support.kan import KAN
from GNN_Layer.GNN import GCN_Layer, GAT_Layer
from model2.moduel.ST_blocks import ST_block
from model.model_baselines import AdaptiveGCN
from _Support.GraphKAN import kanChebConv
import math
from GNN_Layer.DynamicGraphConv import DynamicGraphConv


# 跨尺度注意力机制
class CrossScaleAttention(nn.Module):
    def __init__(self, d_model, n_heads, T_source, T_target, dropout=0.1):
        super(CrossScaleAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.align_conv = nn.Conv2d(
            in_channels = T_source,
            out_channels = T_target,
            kernel_size = (1, 1),
        )

    def forward(self, target, source):
        """
        target: 目标尺度特征 [B, N, T_t, D]
        source: 跨尺度特征 [B, N, T_s, D]
        """
        B, N, T_t, D = target.shape

        # LayerNorm 预归一化
        target_norm = self.layer_norm(target)
        source_norm = self.layer_norm(source)

        # 时间维度对齐
        source_resized = self.align_conv(source_norm.permute(0, 2, 1, 3)) # [B, T_t, N, D]
        source_resized = source_resized.permute(0, 2, 1, 3)  # [B, N, T_t, D]

        # 多头注意力
        Q = self.W_Q(target_norm).view(B, N, T_t, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(source_resized).view(B, N, T_t, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(source_resized).view(B, N, T_t, self.n_heads, self.d_k).transpose(2, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # 对注意力权重加 Dropout

        out = torch.matmul(attn, V)
        out = out.transpose(2, 3).contiguous().view(B, N, T_t, D)

        out = self.out_proj(out)
        out = self.dropout(out)  # 对输出加 Dropout

        # 残差连接
        out = out + target
        return out


class BiDirectionalPyramidFusion(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, down_sampling_window, down_sampling_layers, dropout=0.1):
        super(BiDirectionalPyramidFusion, self).__init__()
        # 季节项 自底向上
        self.season_bottom_up_modules = nn.ModuleList([
            CrossScaleAttention(d_model, n_heads, seq_len // (down_sampling_window ** i),
                                seq_len // (down_sampling_window ** (i + 1)), dropout)
            for i in range(down_sampling_layers)
        ])

        # 季节项 自顶向下
        self.season_top_down_modules = nn.ModuleList([
            CrossScaleAttention(d_model, n_heads, seq_len // (down_sampling_window ** (i + 1)),
                                seq_len // (down_sampling_window ** i), dropout)
            for i in range(down_sampling_layers)
        ])

        # 趋势项 自底向上
        self.trend_bottom_up_modules = nn.ModuleList([
            CrossScaleAttention(d_model, n_heads, seq_len // (down_sampling_window ** i),
                                seq_len // (down_sampling_window ** (i + 1)), dropout)
            for i in range(down_sampling_layers)
        ])

        # 趋势项 自顶向下
        self.trend_top_down_modules = nn.ModuleList([
            CrossScaleAttention(d_model, n_heads, seq_len // (down_sampling_window ** (i + 1)),
                                seq_len // (down_sampling_window ** i), dropout)
            for i in range(down_sampling_layers)
        ])

        # 添加 MLP-Aggregate 模块（用于跨层融合）
        self.season_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
            ) for _ in range(down_sampling_layers + 1)
        ])

        self.trend_aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
            ) for _ in range(down_sampling_layers + 1)
        ])

    def forward(self, season_list, trend_list):
        L = len(season_list)

        # --------- Season Bottom-Up ---------
        season_bottom_up = [season_list[0]]
        prev = season_list[0]
        for i in range(1, L):
            fused = self.season_bottom_up_modules[i - 1](season_list[i], prev)
            season_bottom_up.append(fused)
            prev = fused

        # --------- Season Top-Down ---------
        season_top_down = [None] * L
        season_top_down[-1] = season_list[-1]
        prev = season_list[-1]
        for i in range(L - 2, -1, -1):
            fused = self.season_top_down_modules[i](season_list[i], prev)
            season_top_down[i] = fused
            prev = fused

        # --------- Trend Bottom-Up ---------
        trend_bottom_up = [trend_list[0]]
        prev = trend_list[0]
        for i in range(1, L):
            fused = self.trend_bottom_up_modules[i - 1](trend_list[i], prev)
            trend_bottom_up.append(fused)
            prev = fused

        # --------- Trend Top-Down ---------
        trend_top_down = [None] * L
        trend_top_down[-1] = trend_list[-1]
        prev = trend_list[-1]
        for i in range(L - 2, -1, -1):
            fused = self.trend_top_down_modules[i](trend_list[i], prev)
            trend_top_down[i] = fused
            prev = fused

        # --------- 融合 Bottom-Up 和 Top-Down via MLP-Aggregate ---------
        season_fused = [
            self.season_aggregators[i](season_bottom_up[i] + season_top_down[i])
            for i in range(L)
        ]
        trend_fused = [
            self.trend_aggregators[i](trend_bottom_up[i] + trend_top_down[i])
            for i in range(L)
        ]

        return season_fused, trend_fused


class CrossScaleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, source_len, target_len):
        super(CrossScaleMLP, self).__init__()
        self.mlp1 = nn.Linear(source_len, target_len)
        self.mlp2 = nn.Linear(target_len, target_len)
        self.gelu = nn.GELU()

    def forward(self, target, source):
        """
        :param x: [B, N, T, F]
        :return: [B, N, T, F]
        """
        out = source.permute(0, 1, 3, 2)
        out = self.mlp1(out)
        out = self.gelu(out)
        out = self.mlp2(out)
        out = out.permute(0, 1, 3, 2)

        return out + target


class MultiScaleFusionModule(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, down_sampling_layers, down_sampling_window):
        super(MultiScaleFusionModule, self).__init__()
        # 季节项自底向上
        self.season_bottom_up_modules = nn.ModuleList([
            CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                                seq_len // (down_sampling_window ** (i + 1)))
            for i in range(down_sampling_layers)
        ])

        # 趋势项自顶向下
        self.trend_top_down_modules = nn.ModuleList([
            CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                                seq_len // (down_sampling_window ** i))
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list, trend_list):
        L = len(season_list)

        # --------- Season Bottom-Up ---------
        season_bottom_up = [season_list[0]]
        prev = season_list[0]
        for i in range(1, L):
            fused = self.season_bottom_up_modules[i - 1](season_list[i], prev)
            season_bottom_up.append(fused)
            prev = fused

        # --------- Trend Top-Down ---------
        trend_top_down = [None] * L
        trend_top_down[-1] = trend_list[-1]
        prev = trend_list[-1]
        for i in range(L - 2, -1, -1):
            fused = self.trend_top_down_modules[i](trend_list[i], prev)
            trend_top_down[i] = fused
            prev = fused

        return [season_bottom_up[i] + trend_top_down[i] for i in range(L)]


class MultiScaleFusionModule_Alation(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, down_sampling_layers, down_sampling_window, season_bottom_up_bool, trend_top_down_bool):
        super(MultiScaleFusionModule_Alation, self).__init__()
        self.season_bottom_up_bool = season_bottom_up_bool
        self.trend_top_down_bool = trend_top_down_bool

        if season_bottom_up_bool:
            # 季节项自底向上
            self.season_bottom_up_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                                    seq_len // (down_sampling_window ** (i + 1)))
                for i in range(down_sampling_layers)
            ])

        if trend_top_down_bool:
            # 趋势项自顶向下
            self.trend_top_down_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                                    seq_len // (down_sampling_window ** i))
                for i in range(down_sampling_layers)
            ])

    def forward(self, season_list, trend_list):
        L = len(season_list)

        if self.season_bottom_up_bool:
            # --------- Season Bottom-Up ---------
            season_bottom_up = [season_list[0]]
            prev = season_list[0]
            for i in range(1, L):
                fused = self.season_bottom_up_modules[i - 1](season_list[i], prev)
                season_bottom_up.append(fused)
                prev = fused
            return [season_bottom_up[i] + trend_list[i] for i in range(L)]

        if self.trend_top_down_bool:
            # --------- Trend Top-Down ---------
            trend_top_down = [None] * L
            trend_top_down[-1] = trend_list[-1]
            prev = trend_list[-1]
            for i in range(L - 2, -1, -1):
                fused = self.trend_top_down_modules[i](trend_list[i], prev)
                trend_top_down[i] = fused
                prev = fused
            return [season_list[i] + trend_top_down[i] for i in range(L)]

        # return [season_bottom_up[i] + trend_top_down[i] for i in range(L)]

class MultiScaleFusionModule_Alation_2(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, down_sampling_layers, down_sampling_window):
        super(MultiScaleFusionModule_Alation_2, self).__init__()
        # 季节项自底向上
        self.season_bottom_up_modules = nn.ModuleList([
            CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                                seq_len // (down_sampling_window ** (i + 1)))
            for i in range(down_sampling_layers)
        ])

        # 趋势项自顶向下
        self.trend_top_down_modules = nn.ModuleList([
            CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                                seq_len // (down_sampling_window ** i))
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list, trend_list):
        L = len(season_list)

        # --------- Trend Bottom-Up ---------
        season_bottom_up = [trend_list[0]]
        prev = trend_list[0]
        for i in range(1, L):
            fused = self.season_bottom_up_modules[i - 1](trend_list[i], prev)
            season_bottom_up.append(fused)
            prev = fused

        # --------- Season Top-Down ---------
        trend_top_down = [None] * L
        trend_top_down[-1] = season_list[-1]
        prev = season_list[-1]
        for i in range(L - 2, -1, -1):
            fused = self.trend_top_down_modules[i](season_list[i], prev)
            trend_top_down[i] = fused
            prev = fused

        return [season_bottom_up[i] + trend_top_down[i] for i in range(L)]

class MultiScaleFusionModule_Alation_3(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, down_sampling_layers, down_sampling_window, bottom_up_bool, top_down_bool):
        super(MultiScaleFusionModule_Alation_3, self).__init__()
        self.bottom_up_bool = bottom_up_bool
        self.top_down_bool = top_down_bool

        if self.bottom_up_bool:
            # 季节项自底向上
            self.season_bottom_up_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                                    seq_len // (down_sampling_window ** (i + 1)))
                for i in range(down_sampling_layers)
            ])

            self.trend_bottom_up_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                              seq_len // (down_sampling_window ** (i + 1)))
                for i in range(down_sampling_layers)
            ])

        if self.top_down_bool:
            # 趋势项自顶向下
            self.season_top_down_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                              seq_len // (down_sampling_window ** i))
                for i in range(down_sampling_layers)
            ])

            # 趋势项自顶向下
            self.trend_top_down_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                                    seq_len // (down_sampling_window ** i))
                for i in range(down_sampling_layers)
            ])

    def forward(self, season_list, trend_list):
        L = len(season_list)

        if self.bottom_up_bool:
            # ---------Season Bottom-Up ---------
            season_bottom_up = [season_list[0]]
            prev_season = season_list[0]
            for i in range(1, L):
                fused = self.season_bottom_up_modules[i - 1](season_list[i], prev_season)
                season_bottom_up.append(fused)
                prev_season = fused
            # ---------Trend Bottom-Up ---------
            trend_bottom_up = [trend_list[0]]
            prev_trend = trend_list[0]
            for i in range(1, L):
                fused = self.trend_bottom_up_modules[i - 1](trend_list[i], prev_trend)
                trend_bottom_up.append(fused)
                prev_trend = fused

            return [season_bottom_up[i] + trend_bottom_up[i] for i in range(L)]

        if self.top_down_bool:
            # --------- Trend Top-Down ---------
            season_top_down = [None] * L
            season_top_down[-1] = season_list[-1]
            prev_season = season_list[-1]
            for i in range(L - 2, -1, -1):
                fused = self.season_top_down_modules[i](season_list[i], prev_season)
                season_top_down[i] = fused
                prev_season = fused

            # --------- Trend Top-Down ---------
            trend_top_down = [None] * L
            trend_top_down[-1] = trend_list[-1]
            prev_trend = trend_list[-1]
            for i in range(L - 2, -1, -1):
                fused = self.trend_top_down_modules[i](trend_list[i], prev_trend)
                trend_top_down[i] = fused
                prev_trend = fused
            return [season_top_down[i] + trend_top_down[i] for i in range(L)]


class MultiScaleFusionModule_Alation_wo_Decompose(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, down_sampling_layers, down_sampling_window, season_bottom_up_bool, trend_top_down_bool):
        super(MultiScaleFusionModule_Alation_wo_Decompose, self).__init__()
        self.season_bottom_up_bool = season_bottom_up_bool
        self.trend_top_down_bool = trend_top_down_bool

        if season_bottom_up_bool:
            # 季节项自底向上
            self.season_bottom_up_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** i),
                                    seq_len // (down_sampling_window ** (i + 1)))
                for i in range(down_sampling_layers)
            ])

        if trend_top_down_bool:
            # 趋势项自顶向下
            self.trend_top_down_modules = nn.ModuleList([
                CrossScaleMLP(input_size, hidden_size, seq_len // (down_sampling_window ** (i + 1)),
                                    seq_len // (down_sampling_window ** i))
                for i in range(down_sampling_layers)
            ])

    def forward(self, x_list):
        L = len(x_list)

        if self.season_bottom_up_bool:
            # --------- Season Bottom-Up ---------
            season_bottom_up = [x_list[0]]
            prev = x_list[0]
            for i in range(1, L):
                fused = self.season_bottom_up_modules[i - 1](x_list[i], prev)
                season_bottom_up.append(fused)
                prev = fused
            return season_bottom_up

        if self.trend_top_down_bool:
            # --------- Trend Top-Down ---------
            trend_top_down = [None] * L
            trend_top_down[-1] = x_list[-1]
            prev = x_list[-1]
            for i in range(L - 2, -1, -1):
                fused = self.trend_top_down_modules[i](x_list[i], prev)
                trend_top_down[i] = fused
                prev = fused
            return trend_top_down


class moving_avg_2d(nn.Module):
    """
    Moving average block to highlight the trend of time series
    for input [B, N, T, F], smoothing only along the time dimension
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg_2d, self).__init__()
        self.kernel_size = kernel_size
        # kernel_size=(kernel_size, 1) 表示只在时间维做平滑
        self.avg = nn.AvgPool2d(kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(0, 0))

    def forward(self, x):
        # x: [B, N, T, F]
        # out: [B, N, T, F]

        # 在时间维做 padding
        front = x[:, :, 0:1, :].repeat(1, 1, (self.kernel_size - 1) // 2, 1)
        end = x[:, :, -1:, :].repeat(1, 1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=2)  # [B, N, T + k - 1, F]

        # 调整为 [B, F, T, N] 以适配 AvgPool2d 输入 [B, C, H, W]
        x = x.permute(0, 3, 2, 1)  # B, F, T, N

        # 在时间维做平均池化
        x = self.avg(x)  # B, F, T, N

        # 转回 [B, N, T, F]
        x = x.permute(0, 3, 2, 1)
        return x


# 使用移动平均分解趋势项和季节项
class series_decomposition_2d(nn.Module):
    """
    Series decomposition block for [B, N, T, F] using AvgPool2d
    """

    def __init__(self, kernel_size):
        super(series_decomposition_2d, self).__init__()
        self.moving_avg = moving_avg_2d(kernel_size, stride=1)

    def forward(self, x):
        # x: [B, N, T, F]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Attention_GCN_BiMamba(nn.Module):
    def __init__(self, device, in_channels, out_channels, num_nodes, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, ST_Attention_bool=True):
        super(Attention_GCN_BiMamba, self).__init__()
        # 创建多个ST_block
        self.ST_blocks = nn.ModuleList(
            [ST_block(device, in_channels if i == 0 else out_channels, out_channels, num_nodes, T_in, K, adj_matrix, config, dropout, ST_Attention_bool)
             for i in range(num_of_blocks)]) # [B, N, F_out, T]

    def forward(self, x):
        """
        :param x: [B, N, T_in, F_in]
        :return: [B, N, T_in, hidden_size]
        """
        x = x.transpose(2, 3)  # [B, N, F_in, T_in]
        for block in self.ST_blocks:
            x = block(x)  # [B, N, F_out, T_in]
        x = x.transpose(2, 3)  # [B, N, T_in, F_out]
        return x


class DynamicGraphConv_BiMamba(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(DynamicGraphConv_BiMamba, self).__init__()
        self.res_linear = nn.Linear(in_channels, hidden_size)
        self.gcn1 = DynamicGraphConv(in_channels, hidden_size, hidden_size)
        self.gcn2 = DynamicGraphConv(hidden_size, hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        # self.BiMamba = BidirectionalMamba(hidden_size, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_in, D]
        """
        B, N, t, d = x.size()
        out = []
        for i in range(t):
            x_t = self.gcn1(x[:, :, i, :])
            x_t = self.gcn2(x_t)
            out.append(x_t)
        out = torch.stack(out, dim=2)
        out = F.relu(out)
        # out = out.reshape(B * N, t, -1)
        # out = self.BiMamba(out)
        # out = out.reshape(B, N, t, -1)
        # res = self.res_linear(x)
        # out = out + res
        # out = self.layer_norm(out)
        return out


class GC_LSTM(nn.Module):
    def __init__(self, adj, in_channels, hidden_size, device, num_layers, dropout):
        super(GC_LSTM, self).__init__()
        self.gcn1 = GCN_Layer(device, adj, in_channels, hidden_size)
        self.gcn2 = GCN_Layer(device, adj, hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_in, D]
        """
        B, N, t, d = x.size()
        x = x.permute(0, 2, 1, 3)
        out = x.reshape(B*t, N, d)
        out = self.gcn1(out)
        out = F.relu(out)
        out = self.gcn2(out)
        out = F.relu(out)
        out = out.reshape(B, t, N, -1)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        return out


class AGCLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout):
        super(AGCLSTM, self).__init__()
        self.agcn = AdaptiveGCN(in_channels, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out]
        """
        B, N, t, d = x.size()
        out = []
        for i in range(t):
            out.append(self.agcn(x[:, :, i, :]))
        out = torch.stack(out, dim=2)
        out = out.reshape(B*N, t, -1)
        out, _ = self.lstm(out)
        out = out.reshape(B, N, t, -1)
        return out


# class M_GCN(nn.Module):
#     def __init__(self, device, adj_distance, in_channels, hidden_size, adaptive_adj_bool, distance_bool, features_adj):
#         super(M_GCN, self).__init__()
#         self.adaptive_adj_bool = adaptive_adj_bool
#         self.distance_bool = distance_bool
#
#         self.agcn = AdaptiveGCN(in_channels, hidden_size)
#         self.agcn = DynamicGraphConv(in_channels, hidden_size, hidden_size)
#         self.gcn = GCN_Layer(device, adj_distance, in_channels, hidden_size)
#
#         self.f_agcn = nn.Linear(hidden_size, hidden_size)
#         self.f_graph1 = nn.Linear(hidden_size, hidden_size)

    # def forward(self, x):
    #     """
    #     :param x: [B, N, T_in, D]
    #     :return: [B, N, T_out]
    #     """
    #     B, N, t, d = x.size()
    #     agcn_out = []
    #     for i in range(t):
    #         agcn_out.append(self.agcn(x[:, :, i, :]))
    #     agcn_out = torch.stack(agcn_out, dim=2)
    #
    #     gcn_input = x.permute(0, 2, 1, 3).reshape(B*t, N, d)
    #
    #     gcn_out = self.gcn(gcn_input)
    #     gcn_out = gcn_out.reshape(B, t, N, -1).permute(0, 2, 1, 3)
    #
    #     gate = torch.sigmoid(self.f_agcn(agcn_out) + self.f_graph1(gcn_out))
    #     gate_out = agcn_out * gate + gcn_out * (1 - gate)
    #
    #     return gate_out

    # def forward(self, x):
    #     """
    #     :param x: [B, N, T_in, D]
    #     :return: [B, N, T_out, hidden_size]
    #     """
    #     B, N, t, d = x.size()
    #
    #     # AGCN输出
    #     if self.adaptive_adj_bool:
    #         agcn_out = []
    #         for i in range(t):
    #             agcn_out.append(self.agcn(x[:, :, i, :]))
    #         agcn_out = torch.stack(agcn_out, dim=2)
    #     else:
    #         agcn_out = None
    #
    #     # 基于距离的GCN输出
    #     if self.distance_bool:
    #         gcn_input = x.permute(0, 2, 1, 3).reshape(B * t, N, d)
    #         gcn_out = self.gcn(gcn_input)
    #         gcn_out = gcn_out.reshape(B, t, N, -1).permute(0, 2, 1, 3)
    #     else:
    #         gcn_out = None
    #
    #     # 根据可用输出进行融合
    #     if agcn_out is not None and gcn_out is not None:
    #         gate = torch.sigmoid(self.f_agcn(agcn_out) + self.f_graph1(gcn_out))
    #         gate_out = agcn_out * gate + gcn_out * (1 - gate)
    #     elif agcn_out is not None:
    #         gate_out = agcn_out
    #     elif gcn_out is not None:
    #         gate_out = gcn_out
    #     else:
    #         raise ValueError("至少启用一个GCN模块")
    #
    #     return gate_out


class M_GCN(nn.Module):
    def __init__(self, device, adj_distance, in_channels, hidden_size, adaptive_adj_bool=False, distance_bool=False, features_adj=False):
        super(M_GCN, self).__init__()

        # 检查只有一个为 True
        flags = [adaptive_adj_bool, distance_bool, features_adj]
        if sum(flags) != 1:
            raise ValueError("adaptive_adj_bool, distance_bool, features_adj 必须且只能有一个为 True")

        self.adaptive_adj_bool = adaptive_adj_bool
        self.distance_bool = distance_bool
        self.features_adj = features_adj

        # 根据标志初始化相应模块
        if self.adaptive_adj_bool:
            self.gcn_module = DynamicGraphConv(in_channels, hidden_size, hidden_size)
        elif self.distance_bool:
            self.gcn_module = nn.Sequential(
                # GCN_Layer(device, adj_distance, in_channels, hidden_size),
                # nn.ReLU(),
                # GCN_Layer(device, adj_distance, hidden_size, hidden_size),
                # nn.ReLU(),
                GAT_Layer(device, adj_distance, in_channels, hidden_size, edge_dim=1),
                nn.ReLU(),
            )
        elif self.features_adj:
            self.gcn_module = AdaptiveGCN(in_channels, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        B, N, t, d = x.size()

        if self.distance_bool:
            gcn_input = x.permute(0, 2, 1, 3).reshape(B * t, N, d)  # [B*T, N, D]
            out  = self.gcn_module(gcn_input)  # [B*T, N, hidden_size]
            out = out.reshape(B, t, N, -1).permute(0, 2, 1, 3)  # [B, N, T, hidden_size]
        else:
            out = []
            for i in range(t):
                out.append(self.gcn_module(x[:, :, i, :]))  # [B, N, hidden_size]
            out = torch.stack(out, dim=2)  # [B, N, T_in, hidden_size]
        return out



class M_GraphKAN(nn.Module):
    # def __init__(self, in_channels, hidden_size, K, edge_index_distance, edge_weight_distance, edge_index_DTW, edge_weight_DTW):
    def __init__(self, in_channels, hidden_size, K, edge_index_distance, edge_weight_distance):
        super(M_GraphKAN, self).__init__()
        self.edge_index_distance = edge_index_distance
        self.edge_weight_distance = edge_weight_distance
        # self.edge_index_DTW = edge_index_DTW
        # self.edge_weight_DTW = edge_weight_DTW

        self.agcn = AdaptiveGCN(in_channels, hidden_size)
        self.graphkan1 = kanChebConv(in_channels, hidden_size, K)
        # self.graphkan2 = kanChebConv(hidden_size, hidden_size, K)

        # self.graphkan = GraphKAN(in_channels, hidden_size, hidden_size)

        self.f_agcn = nn.Linear(hidden_size, hidden_size)
        self.f_graphkan1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_in, D]
        """
        B, N, t, d = x.size()
        agcn_out = []
        graphkan1_out = []
        # graphkan_out = []
        # graphkan2_out = []
        for i in range(t):
            agcn_out.append(self.agcn(x[:, :, i, :]))
            graphkan1_out.append(self.graphkan1(x[:, :, i, :], self.edge_index_distance, self.edge_weight_distance))
            # graphkan2_out.append(self.graphkan2(x[:, :, i, :], self.edge_index_DTW, self.edge_weight_DTW))
            # graphkan_out = self.graphkan(x[:, :, i, :], self.edge_index_distance, self.edge_weight_distance)
        agcn_out = torch.stack(agcn_out, dim=2)
        graphkan1_out = torch.stack(graphkan1_out, dim=2)
        # graphkan2_out = torch.stack(graphkan2_out, dim=2)
        # graphkan_out = torch.stack(graphkan_out, dim=2)

        # gcn_out = agcn_out + graphkan1_out + graphkan2_out
        # gcn_out = agcn_out + graphkan1_out

        # gate
        gate = torch.sigmoid(self.f_agcn(agcn_out) + self.f_graphkan1(graphkan1_out))
        # gate = torch.sigmoid(self.f_agcn(agcn_out) + self.f_graphkan1(graphkan_out))
        gcn_out = gate * agcn_out + (1 - gate) * graphkan1_out
        # gcn_out = gate * agcn_out + (1 - gate) * graphkan_out

        return gcn_out


# 该分解方法用不到
class DFT_series_decomposition(nn.Module):
    """
    使用傅里叶变换将时间序列分解为趋势项和季节项
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomposition, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        """
        输入:
            x: [B, N, T, F] 张量
        输出:
            x_trend: 趋势项 [B, N, T, F]
            x_season: 季节项 [B, N, T, F]
        """
        B, N, T, F = x.shape

        # 1. 对时间维度做实数快速傅里叶变换
        xf = torch.fft.rfft(x, dim=2)  # [B, N, T//2 + 1, F]

        # 2. 计算频率的幅值
        freq = torch.abs(xf)
        freq[..., 0, :] = 0  # 去掉直流分量

        # 3. 选取前 top_k 个频率
        top_k_freq, _ = torch.topk(freq, k=self.top_k, dim=2)
        threshold = top_k_freq.min(dim=2, keepdim=True).values

        # 4. 保留主频得到季节项
        xf_season = xf.clone()
        xf_season[freq < threshold] = 0
        x_season = torch.fft.irfft(xf_season, n=T, dim=2)

        # 5. 趋势项 = 原序列 - 季节项
        x_trend = x - x_season

        return x_trend, x_season


class MultiScaleSeasonMixing(nn.Module): # 自下而上
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        kernel_size=(1, 1)
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        kernel_size=(1, 1)
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        """
        :param season_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # mixing high->low
        out_high = season_list[0].permute(0, 2, 1, 3) # [B, T, N, F]
        out_low = season_list[1].permute(0, 2, 1, 3) # [B, T, N, F]
        out_season_list = [out_high.permute(0, 2, 1, 3)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high) # 时间维度下采样
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2].permute(0, 2, 1, 3)
            out_season_list.append(out_high.permute(0, 2, 1, 3))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(  # 线性层 -> GELU激活函数 -> 线性层
                    nn.Conv2d(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                        kernel_size=(1, 1)
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                        kernel_size=(1, 1)
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        """
        :param trend_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0].permute(0, 2, 1, 3)
        out_high = trend_list_reverse[1].permute(0, 2, 1, 3)
        out_trend_list = [out_low.permute(0, 2, 1, 3)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2].permute(0, 2, 1, 3)
            out_trend_list.append(out_low.permute(0, 2, 1, 3))

        out_trend_list.reverse()
        return out_trend_list

# KAN based MultiScaleSeasonMixing
# class MultiScaleSeasonMixing(nn.Module): # 自下而上
#     """
#     Bottom-up mixing season pattern
#     """
#
#     def __init__(self, configs):
#         super(MultiScaleSeasonMixing, self).__init__()
#
#         self.down_sampling_layers = torch.nn.ModuleList(
#             [
#                 nn.Sequential( # KAN -> GELU -> KAN
#                     KAN([configs.seq_len // (configs.down_sampling_window ** i),
#                          # configs.d_model,
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),]),
#                     nn.GELU(),
#                     KAN([configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                          # configs.d_model,
#                          configs.seq_len // (configs.down_sampling_window ** (i + 1)),])
#
#                 )
#                 for i in range(configs.down_sampling_layers)
#             ]
#         )
#
#     def forward(self, season_list):
#         """
#         :param season_list: list of tensor shaped of [B, N, T, F]
#         :return: list of tensor shaped of [B, N, T, F]
#         """
#         # mixing high->low
#         out_high = season_list[0].permute(0, 1, 3, 2) # [B, T, N, F]
#         out_low = season_list[1].permute(0, 1, 3, 2) # [B, T, N, F]
#         out_season_list = [out_high.permute(0, 1, 3, 2)]
#
#         for i in range(len(season_list) - 1):
#             out_low_res = self.down_sampling_layers[i](out_high) # 时间维度下采样
#             out_low = out_low + out_low_res
#             out_high = out_low
#             if i + 2 <= len(season_list) - 1:
#                 out_low = season_list[i + 2].permute(0, 1, 3, 2)
#             out_season_list.append(out_high.permute(0, 1, 3, 2))
#
#         return out_season_list

# KAN based MultiScaleTrendMixing
# class MultiScaleTrendMixing(nn.Module):
#     """
#     Top-down mixing trend pattern
#     """
#     def __init__(self, configs):
#         super(MultiScaleTrendMixing, self).__init__()
#         self.up_sampling_layers = torch.nn.ModuleList(
#             [
#                 nn.Sequential(  # KAN -> GELU激活函数 -> KAN
#                     KAN([
#                         configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                         # configs.d_model,
#                         configs.seq_len // (configs.down_sampling_window ** i)
#                     ]),
#                     nn.GELU(),
#                     KAN([
#                         configs.seq_len // (configs.down_sampling_window ** i),
#                         # configs.d_model,
#                         configs.seq_len // (configs.down_sampling_window ** i)
#                     ])
#                 )
#                 for i in reversed(range(configs.down_sampling_layers))
#             ])
#
#     def forward(self, trend_list):
#         """
#         :param trend_list: list of tensor shaped of [B, N, T, F]
#         :return: list of tensor shaped of [B, N, T, F]
#         """
#         # mixing low->high
#         trend_list_reverse = trend_list.copy()
#         trend_list_reverse.reverse()
#         out_low = trend_list_reverse[0].permute(0, 1, 3, 2)
#         out_high = trend_list_reverse[1].permute(0, 1, 3, 2)
#         out_trend_list = [out_low.permute(0, 1, 3, 2)]
#
#         for i in range(len(trend_list_reverse) - 1):
#             out_high_res = self.up_sampling_layers[i](out_low)
#             out_high = out_high + out_high_res
#             out_low = out_high
#             if i + 2 <= len(trend_list_reverse) - 1:
#                 out_high = trend_list_reverse[i + 2].permute(0, 1, 3, 2)
#             out_trend_list.append(out_low.permute(0, 1, 3, 2))
#
#         out_trend_list.reverse()
#         return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs, down_sampling_layers):
        super(PastDecomposableMixing, self).__init__()
        self.down_sampling_layers =down_sampling_layers
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 单向跨尺度融合
        # self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        # self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # Bidirectional pyramid fusion
        # 双向跨尺度融合
        # self.mixing_multi_scale_season_trend = BiDirectionalPyramidFusion(configs.seq_len, configs.d_model, configs.n_heads,
        #                                         configs.down_sampling_window, configs.down_sampling_layers, configs.dropout)


        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule(configs.d_model, configs.d_model, configs.seq_len,
                                                         self.down_sampling_layers,
                                                         configs.down_sampling_window)

        # self.mlp_aggregate = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.GELU(),
        # )

    def forward(self, x_list):
        """
        :param x_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season)
            trend_list.append(trend)

        # 单向跨尺度融合
        # bottom-up season mixing
        # out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        # out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # out_list = []
        # for out_season, out_trend in zip(out_season_list, out_trend_list):
        #     out = out_season + out_trend
        #     out_list.append(out)

        # out_list = []
        # for x, out_season, out_trend in zip(x_list, out_season_list, out_trend_list):
        #     out = x + out_season + out_trend
        #     out_list.append(out)

        # 双向跨尺度融合
        # out_season_list, out_trend_list = self.mixing_multi_scale_season_trend(season_list, trend_list)
        #
        # out_list = []
        # for x, s, t in zip(x_list, out_season_list, out_trend_list):
        #     fused = self.mlp_aggregate(s + t)
        #     out = x + fused
        #     out_list.append(out)

        out_list = self.multi_scale_fusion(season_list, trend_list)

        return out_list


class PastDecomposableMixing_Alation(nn.Module):
    def __init__(self, configs, down_sampling_layers, season_bottom_up_bool, trend_top_down_bool):
        super(PastDecomposableMixing_Alation, self).__init__()
        self.down_sampling_layers =down_sampling_layers
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 单向跨尺度融合
        # self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        # self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # Bidirectional pyramid fusion
        # 双向跨尺度融合
        # self.mixing_multi_scale_season_trend = BiDirectionalPyramidFusion(configs.seq_len, configs.d_model, configs.n_heads,
        #                                         configs.down_sampling_window, configs.down_sampling_layers, configs.dropout)

        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule_Alation(configs.d_model, configs.d_model, configs.seq_len,
                                                         self.down_sampling_layers,
                                                         configs.down_sampling_window, season_bottom_up_bool, trend_top_down_bool)

        # self.mlp_aggregate = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.GELU(),
        # )

    def forward(self, x_list):
        """
        :param x_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season)
            trend_list.append(trend)

        # 单向跨尺度融合
        # bottom-up season mixing
        # out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        # out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # out_list = []
        # for out_season, out_trend in zip(out_season_list, out_trend_list):
        #     out = out_season + out_trend
        #     out_list.append(out)

        # out_list = []
        # for x, out_season, out_trend in zip(x_list, out_season_list, out_trend_list):
        #     out = x + out_season + out_trend
        #     out_list.append(out)

        # 双向跨尺度融合
        # out_season_list, out_trend_list = self.mixing_multi_scale_season_trend(season_list, trend_list)
        #
        # out_list = []
        # for x, s, t in zip(x_list, out_season_list, out_trend_list):
        #     fused = self.mlp_aggregate(s + t)
        #     out = x + fused
        #     out_list.append(out)

        out_list = self.multi_scale_fusion(season_list, trend_list)

        return out_list


class PastDecomposableMixing_Alation_2(nn.Module):
    def __init__(self, configs, down_sampling_layers):
        super(PastDecomposableMixing_Alation_2, self).__init__()
        self.down_sampling_layers =down_sampling_layers
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule_Alation_2(configs.d_model, configs.d_model, configs.seq_len,
                                                         self.down_sampling_layers,
                                                         configs.down_sampling_window)

    def forward(self, x_list):
        """
        :param x_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season)
            trend_list.append(trend)

        out_list = self.multi_scale_fusion(season_list, trend_list)

        return out_list

class PastDecomposableMixing_Alation_3(nn.Module):
    def __init__(self, configs, down_sampling_layers, bottom_up_bool, top_down_bool):
        super(PastDecomposableMixing_Alation_3, self).__init__()
        self.down_sampling_layers =down_sampling_layers
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 单向跨尺度融合
        # self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        # self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        # Bidirectional pyramid fusion
        # 双向跨尺度融合
        # self.mixing_multi_scale_season_trend = BiDirectionalPyramidFusion(configs.seq_len, configs.d_model, configs.n_heads,
        #                                         configs.down_sampling_window, configs.down_sampling_layers, configs.dropout)


        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule_Alation_3(configs.d_model, configs.d_model, configs.seq_len,
                                                         self.down_sampling_layers,
                                                         configs.down_sampling_window, bottom_up_bool, top_down_bool)

        # self.mlp_aggregate = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model),
        #     nn.GELU(),
        # )

    def forward(self, x_list):
        """
        :param x_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season)
            trend_list.append(trend)

        # 单向跨尺度融合
        # bottom-up season mixing
        # out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        # out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # out_list = []
        # for out_season, out_trend in zip(out_season_list, out_trend_list):
        #     out = out_season + out_trend
        #     out_list.append(out)

        # out_list = []
        # for x, out_season, out_trend in zip(x_list, out_season_list, out_trend_list):
        #     out = x + out_season + out_trend
        #     out_list.append(out)

        # 双向跨尺度融合
        # out_season_list, out_trend_list = self.mixing_multi_scale_season_trend(season_list, trend_list)
        #
        # out_list = []
        # for x, s, t in zip(x_list, out_season_list, out_trend_list):
        #     fused = self.mlp_aggregate(s + t)
        #     out = x + fused
        #     out_list.append(out)

        out_list = self.multi_scale_fusion(season_list, trend_list)

        return out_list

class PastDecomposableMixing_Alation_wo_Decompose(nn.Module):
    def __init__(self, configs, down_sampling_layers, season_bottom_up_bool, trend_top_down_bool):
        super(PastDecomposableMixing_Alation_wo_Decompose, self).__init__()
        self.down_sampling_layers =down_sampling_layers

        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule_Alation_wo_Decompose(configs.d_model, configs.d_model, configs.seq_len,
                                                         self.down_sampling_layers,
                                                         configs.down_sampling_window, season_bottom_up_bool, trend_top_down_bool)

    def forward(self, x_list):
        """
        :param x_list: list of tensor shaped of [B, N, T, F]
        :return: list of tensor shaped of [B, N, T, F]
        """
        out_list = self.multi_scale_fusion(x_list)

        return out_list


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(embed_size, 1, 1)

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.conv1(input_prediction_layer) # 等号左边 out shape: [B, T, N, d]
        out = out.permute(0, 3, 2, 1)  # 等号左边 out shape: [B, d, N, T]
        out = self.conv2(out)  # 等号左边 out shape: [B, 1, N, T]
        out = out.squeeze(1)

        return out

class PE(nn.Module):
    def __init__(self, T: int, D: int):
        """
        Args:
            T: 时间序列长度
            D: 特征维度
        """
        super().__init__()
        # 预计算 [1, 1, T, D] 的正弦位置编码
        pe = torch.zeros(T, D)
        position = torch.arange(0, T).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(-math.log(10000.0) * (torch.arange(0, D, 2) / D))  # [D/2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
        self.register_buffer("pe", pe)     # 不作为参数更新，但会随模型保存/加载

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, N, T, D]
        Returns:
            x 加上位置编码 [B, N, T, D]
        """
        return x + self.pe



class Model5(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, num_nodes, num_layers, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool, features_adj_bool):
        super(Model5, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs, self.down_sampling_layers) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        self.STGCN_layers = nn.ModuleList([
            nn.Sequential(
                M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool, features_adj_bool)
            )
            for _ in range(self.down_sampling_layers + 1)
        ])


        # 预测层
        self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
            [
                PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    T_out,
                    out_channels,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list

    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # 预测层一般接受 [B, T, N, F] 输入，转置一下
            dec_out_i = self.predict_layers[i](enc_out.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
            dec_out_list.append(dec_out_i)
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)


        # 对每个尺度时间序列进行时空特征提取
        enc_out_list = []
        for i, x_in in enumerate(x_enc_list):
            out = self.STGCN_layers[i](x_in)  # 经过时空图卷积+归一化+dropout
            enc_out_list.append(out)

        # 跨尺度融合
        for i in range(self.layer): # self.layer代表多尺度特征融合的层数
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 多尺度预测
        dec_out_list = self.future_multi_mixing(enc_out_list)

        # 多尺度融合模块
        dec_out_stack = torch.stack(dec_out_list, dim=-1)  # [B, N, T_out, scales]
        dec_out = dec_out_stack.sum(dim=-1)

        return dec_out

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out

class Model5_Alation_wo_Decompose(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, num_nodes, num_layers, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool, features_adj_bool, season_bottom_up_bool, trend_top_down_bool):
        super(Model5_Alation_wo_Decompose, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.input_embedding = nn.Linear(in_channels, configs.d_model)
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing_Alation_wo_Decompose(configs, self.down_sampling_layers, season_bottom_up_bool, trend_top_down_bool) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        # # 位置编码
        # self.PE = nn.ModuleList([
        #     PE(configs.seq_len // (configs.down_sampling_window ** i), configs.d_model)
        #     for i in range(self.down_sampling_layers + 1)
        # ])

        self.STGCN_layers = nn.ModuleList([
            nn.Sequential(
                M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool, features_adj_bool)
            )
            for _ in range(self.down_sampling_layers + 1)
        ])


        # 预测层
        self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
            [
                PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    T_out,
                    out_channels,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list

    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # 预测层一般接受 [B, T, N, F] 输入，转置一下
            dec_out_i = self.predict_layers[i](enc_out.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
            dec_out_list.append(dec_out_i)
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 输入嵌入, 扩展到隐藏层维度
        # input_embedding = self.input_embedding(x_enc)

        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 位置嵌入
        # x_enc_list = [self.PE[i](x_in) for i, x_in in enumerate(x_enc_list)]

        # 对每个尺度时间序列进行时空特征提取
        enc_out_list = []
        for i, x_in in enumerate(x_enc_list):
            out = self.STGCN_layers[i](x_in)  # 经过时空图卷积+归一化+dropout
            enc_out_list.append(out)

        # 跨尺度融合
        for i in range(self.layer): # self.layer代表多尺度特征融合的层数
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 多尺度预测
        dec_out_list = self.future_multi_mixing(enc_out_list)

        # 多尺度融合模块
        dec_out_stack = torch.stack(dec_out_list, dim=-1)  # [B, N, T_out, scales]
        dec_out = dec_out_stack.sum(dim=-1)

        return dec_out

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out


class Model5_Alation(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, num_nodes, num_layers, T_in, T_out, K, adj_matrix, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool, features_adj_bool, Dynamic_GCN_bool, season_bottom_up_bool, trend_top_down_bool, MSDF_bool, Multi_Predictor_bool):
        super(Model5_Alation, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_embedding = nn.Linear(in_channels, configs.d_model)
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing_Alation(configs, self.down_sampling_layers, season_bottom_up_bool, trend_top_down_bool) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        self.Dynamic_GCN_bool = Dynamic_GCN_bool
        self.season_bottom_up_bool = season_bottom_up_bool
        self.trend_top_down_bool = trend_top_down_bool
        self.MSDF_bool = MSDF_bool
        self.Multi_Predictor_bool = Multi_Predictor_bool

        # 位置编码
        # self.PE = nn.ModuleList([
        #     PE(configs.seq_len // (configs.down_sampling_window ** i), configs.d_model)
        #     for i in range(self.down_sampling_layers + 1)
        # ])

        if Dynamic_GCN_bool:
            self.STGCN_layers = nn.ModuleList([
                nn.Sequential(
                    M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool, features_adj_bool)
                )
                for _ in range(self.down_sampling_layers + 1)
            ])


        if self.Multi_Predictor_bool:
            # 预测层
            self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
                [
                    PredictionLayer(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        T_out,
                        out_channels,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )
        else:
            self.predict_layer = PredictionLayer(
                        configs.seq_len,
                        T_out,
                        out_channels,
                    )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list

    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # 预测层一般接受 [B, T, N, F] 输入，转置一下
            dec_out_i = self.predict_layers[i](enc_out.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
            dec_out_list.append(dec_out_i)
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 输入嵌入, 扩展到隐藏层维度
        # input_embedding = self.input_embedding(x_enc)

        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 位置嵌入
        # x_enc_list = [self.PE[i](x_in) for i, x_in in enumerate(x_enc_list)]

        enc_out_list = []
        if self.Dynamic_GCN_bool:
            # 对每个尺度时间序列进行空间特征提取
            for i, x_in in enumerate(x_enc_list):
                out = self.STGCN_layers[i](x_in)
                enc_out_list.append(out)
        else:
            enc_out_list = [self.input_embedding(i) for i in x_enc_list]

        if not self.MSDF_bool:
            enc_out_list = enc_out_list
        else:
            # 跨尺度融合
            for i in range(self.layer): # self.layer代表多尺度特征融合的层数
                enc_out_list = self.pdm_blocks[i](enc_out_list)

        if self.Multi_Predictor_bool:
            # 多尺度预测
            dec_out_list = self.future_multi_mixing(enc_out_list)
        else:
            dec_out_list = enc_out_list

            max_T = max(x.shape[2] for x in dec_out_list)

            # 对齐时间维：对较短的序列在时间维上补零
            padded_list = []
            for x in dec_out_list:
                pad_len = max_T - x.shape[2]
                if pad_len > 0:
                    # 在时间维 (dim=2) 后面补零
                    x = F.pad(x, (0, 0, 0, pad_len))
                padded_list.append(x)

            dec_out = torch.stack(padded_list, dim=0).sum(dim=0)
            dec_out = self.predict_layer(dec_out.transpose(1, 2))
            return dec_out


        # 多尺度融合模块
        dec_out_stack = torch.stack(dec_out_list, dim=-1)  # [B, N, T_out, scales]
        dec_out = dec_out_stack.sum(dim=-1)

        return dec_out

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out

class Model5_Alation_2(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, num_nodes, num_layers, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool, features_adj_bool):
        super(Model5_Alation_2, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.input_embedding = nn.Linear(in_channels, configs.d_model)
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing_Alation_2(configs, self.down_sampling_layers) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        self.STGCN_layers = nn.ModuleList([
            nn.Sequential(
                M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool, features_adj_bool)
            )
            for _ in range(self.down_sampling_layers + 1)
        ])


        # 预测层
        self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
            [
                PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    T_out,
                    out_channels,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list

    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # 预测层一般接受 [B, T, N, F] 输入，转置一下
            dec_out_i = self.predict_layers[i](enc_out.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
            dec_out_list.append(dec_out_i)
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 对每个尺度时间序列进行时空特征提取
        enc_out_list = []
        for i, x_in in enumerate(x_enc_list):
            out = self.STGCN_layers[i](x_in)  # 经过时空图卷积+归一化+dropout
            enc_out_list.append(out)

        # 跨尺度融合
        for i in range(self.layer): # self.layer代表多尺度特征融合的层数
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 多尺度预测
        dec_out_list = self.future_multi_mixing(enc_out_list)

        # 多尺度融合模块
        dec_out_stack = torch.stack(dec_out_list, dim=-1)  # [B, N, T_out, scales]
        dec_out = dec_out_stack.sum(dim=-1)

        return dec_out

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out


class Model5_Alation_3(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, num_nodes, num_layers, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool, features_adj_bool, bottom_up_bool, top_down_bool):
        super(Model5_Alation_3, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing_Alation_3(configs, self.down_sampling_layers, bottom_up_bool, top_down_bool) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        self.bottom_up_bool = bottom_up_bool
        self.top_down_bool = top_down_bool

        self.STGCN_layers = nn.ModuleList([
            nn.Sequential(
                M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool, features_adj_bool)
            )
            for _ in range(self.down_sampling_layers + 1)
        ])


        # 预测层
        self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
            [
                PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    T_out,
                    out_channels,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list

    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            # 预测层一般接受 [B, T, N, F] 输入，转置一下
            dec_out_i = self.predict_layers[i](enc_out.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
            dec_out_list.append(dec_out_i)
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 对每个尺度时间序列进行时空特征提取
        enc_out_list = []
        for i, x_in in enumerate(x_enc_list):
            out = self.STGCN_layers[i](x_in)  # 经过时空图卷积+归一化+dropout
            enc_out_list.append(out)

        # 跨尺度融合
        for i in range(self.layer): # self.layer代表多尺度特征融合的层数
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 多尺度预测
        dec_out_list = self.future_multi_mixing(enc_out_list)

        # 多尺度融合模块
        dec_out_stack = torch.stack(dec_out_list, dim=-1)  # [B, N, T_out, scales]
        dec_out = dec_out_stack.sum(dim=-1)

        return dec_out

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out



class Model5_single_scale(nn.Module):

    def __init__(self, configs, device, in_channels, out_channels, down_sampling_layers, scale_th, num_nodes, num_layers, T_in, T_out, K, adj_matrix, config, dropout,
                 num_of_blocks, edge_index, edge_weights, adaptive_adj_bool, distance_adj_bool):
        super(Model5_single_scale, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs, self.down_sampling_layers) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers
        self.scale_th = scale_th

        self.STGCN_layer = M_GCN(device, adj_matrix, in_channels, out_channels, adaptive_adj_bool, distance_adj_bool)

        # 预测层
        self.predict_layer = PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** scale_th),
                    T_out,
                    out_channels,
                )

    # 获取多尺度时间序列
    def __multi_scale_process_inputs(self, x_enc):
        """
        多尺度输入处理
        :param x_enc: [B, N, T, F]
        :return: list of tensors [B, N, T_i, F]
        """
        x_enc_list = [x_enc]
        x_enc_curr = x_enc.permute(0, 3, 2, 1)  # 转成 [B, F, T, N] 方便池化

        for _ in range(self.down_sampling_layers):
            x_enc_curr = self.down_pool(x_enc_curr)  # 下采样
            x_enc_down = x_enc_curr.permute(0, 3, 2, 1)  # 转回 [B, N, T, F]
            x_enc_list.append(x_enc_down)

        return x_enc_list[self.scale_th]


    def future_multi_mixing(self, enc_out_list):
        """
        多尺度预测
        :param enc_out_list: list of [B, N, T, F]
        :return: list of [B, N, T_out]
        """
        # 预测层一般接受 [B, T, N, F] 输入，转置一下
        dec_out_list = self.predict_layer(enc_out_list.transpose(1, 2))  # [B, T, N, F] -> [B, T_out, N]
        return dec_out_list

    def forecast(self, x_enc):
        """
        预测接口
        :param x_enc: [B, N, T, F]
        :return: [B, N, T_out]
        """
        # 下采样得到多尺度时间序列数据
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        # 对每个尺度时间序列进行时空特征提取
        enc_out_list = self.STGCN_layer(x_enc_list)

        # 多尺度预测
        dec_out_list = self.future_multi_mixing(enc_out_list)

        # 多尺度融合模块
        return dec_out_list

    def forward(self, x_enc):
        """
        :param x_enc: [B, N, T_in, F]
        :return: [B, N, T_out]
        """
        dec_out = self.forecast(x_enc)
        return dec_out


