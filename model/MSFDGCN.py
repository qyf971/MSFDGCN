import torch
import torch.nn as nn
from GNN_Layer.GNN import GCN_Layer, GAT_Layer
from model.model_baselines import AdaptiveGCN
from GNN_Layer.DynamicGraphConv import DynamicGraphConv, DynamicGCNLayer
import torch.nn.functional as F


class DynamicGCN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(DynamicGCN, self).__init__()
        self.dynamic_gcn = DynamicGraphConv(in_channels, hidden_size, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        _, _, t, _ = x.size()
        out = []
        for i in range(t):
            out.append(self.dynamic_gcn(x[:, :, i, :]))  # [B, N, hidden_size]
        out = torch.stack(out, dim=2)  # [B, N, T_in, hidden_size]
        return out

class AdaptiveGCN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(AdaptiveGCN, self).__init__()
        self.adaptive_gcn = AdaptiveGCN(in_channels, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        _, _, t, _ = x.size()
        out = []
        for i in range(t):
            out.append(self.adaptive_gcn(x[:, :, i, :]))  # [B, N, hidden_size]
        out = torch.stack(out, dim=2)  # [B, N, T_in, hidden_size]
        return out

class GCN(nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        super(GCN, self).__init__()
        self.gcn1 = GCN_Layer(device, adj, in_features, out_features)
        self.gcn2 = GCN_Layer(device, adj, out_features, out_features)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        B, N, t, d = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B * t, N, d)  # [B*T, N, D]
        out = self.gcn1(x)  # [B*T, N, hidden_size]
        out = F.relu(out)
        out = self.gcn2(out)
        out = F.relu(out)
        out = out.reshape(B, t, N, -1).permute(0, 2, 1, 3)  # [B, N, T, hidden_size]
        return out

class GAT(nn.Module):
    def __init__(self, device, adj, in_features, out_features):
        super(GAT, self).__init__()
        self.gat = GAT_Layer(device, adj, in_features, out_features)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        B, N, t, d = x.size()
        gat_input = x.permute(0, 2, 1, 3).reshape(B * t, N, d)
        out  = self.gat(gat_input)
        out = out.reshape(B, t, N, -1).permute(0, 2, 1, 3)
        return out




# class DynamicGCN(nn.Module):
#     def __init__(self, in_channels, hidden_size):
#         super(DynamicGCN, self).__init__()
#
#         # --- 【新增：外部 Dropout】 ---
#         self.dropout = nn.Dropout(p=0.2)
#
#         # --- 【新增：层归一化】 ---
#         # 作用于特征维度 [B, N, T, hidden_size] 的最后一个维度
#         self.layer_norm = nn.LayerNorm(hidden_size)
#
#         # 使用更新后的 DynamicGraphConv
#         self.dynamic_gcn = DynamicGraphConv(in_channels, hidden_size, hidden_size)
#
#     def forward(self, x):
#         """
#         :param x: [B, N, T_in, D]
#         :return: [B, N, T_out, hidden_size]
#         """
#         _, _, t, _ = x.size()
#         out_list = []
#         for i in range(t):
#             # GCN 模块内部已包含残差和 Dropout
#             gcn_output = self.dynamic_gcn(x[:, :, i, :])  # [B, N, hidden_size]
#             out_list.append(gcn_output)
#
#         out = torch.stack(out_list, dim=2)  # [B, N, T_in, hidden_size]
#
#         # --- 【应用：外部 Dropout】 ---
#         out = self.dropout(out)
#
#         # --- 【应用：层归一化】 ---
#         out = self.layer_norm(out)
#
#         return out


# class DynamicGCN(nn.Module):
#     def __init__(self, device, adj_distance, in_channels, hidden_size, adaptive_adj_bool=False, distance_bool=False, features_adj=False):
#         super(DynamicGCN, self).__init__()
#
#         # 检查只有一个为 True
#         flags = [adaptive_adj_bool, distance_bool, features_adj]
#         if sum(flags) != 1:
#             raise ValueError("adaptive_adj_bool, distance_bool, features_adj 必须且只能有一个为 True")
#
#         self.adaptive_adj_bool = adaptive_adj_bool
#         self.distance_bool = distance_bool
#         self.features_adj = features_adj
#
#         self.dropout = nn.Dropout(p=0.2)
#
#         # 根据标志初始化相应模块
#         if self.adaptive_adj_bool:
#             self.gcn_module = DynamicGraphConv(in_channels, hidden_size, hidden_size)
#         elif self.distance_bool:
#             self.gcn_module = nn.Sequential(
#                 # GCN_Layer(device, adj_distance, in_channels, hidden_size),
#                 # nn.ReLU(),
#                 # GCN_Layer(device, adj_distance, hidden_size, hidden_size),
#                 # nn.ReLU(),
#                 GAT_Layer(device, adj_distance, in_channels, hidden_size, edge_dim=1),
#                 nn.ReLU(),
#             )
#         elif self.features_adj:
#             self.gcn_module = AdaptiveGCN(in_channels, hidden_size)
#
#     def forward(self, x):
#         """
#         :param x: [B, N, T_in, D]
#         :return: [B, N, T_out, hidden_size]
#         """
#         B, N, t, d = x.size()
#
#         if self.distance_bool:
#             gcn_input = x.permute(0, 2, 1, 3).reshape(B * t, N, d)  # [B*T, N, D]
#             out  = self.gcn_module(gcn_input)  # [B*T, N, hidden_size]
#             out = out.reshape(B, t, N, -1).permute(0, 2, 1, 3)  # [B, N, T, hidden_size]
#         else:
#             out = []
#             for i in range(t):
#                 out.append(self.gcn_module(x[:, :, i, :]))  # [B, N, hidden_size]
#             out = torch.stack(out, dim=2)  # [B, N, T_in, hidden_size]
#         return out


class CrossScaleMLP(nn.Module):
    def __init__(self, source_len, target_len):
        super(CrossScaleMLP, self).__init__()
        self.linear1 = nn.Linear(source_len, target_len)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(target_len, target_len)

    def forward(self, target, source):
        """
        target: [B, N, T_target, F]
        source: [B, N, T_source, F]
        """
        x = source.permute(0, 1, 3, 2)
        out = self.linear1(x)        # [B, N, F, T_target]
        out = self.gelu(out)
        out = self.linear2(out)      # [B, N, F, T_target]
        out = out.permute(0, 1, 3, 2)
        return out + target


class MultiScaleFusionModule(nn.Module):
    def __init__(self, seq_len, down_sampling_layers, down_sampling_window):
        super(MultiScaleFusionModule, self).__init__()
        # 季节项自底向上
        self.season_bottom_up_modules = nn.ModuleList([
            CrossScaleMLP(seq_len // (down_sampling_window ** i), seq_len // (down_sampling_window ** (i + 1)))
            for i in range(down_sampling_layers)
        ])

        # 趋势项自顶向下
        self.trend_top_down_modules = nn.ModuleList([
            CrossScaleMLP(seq_len // (down_sampling_window ** (i + 1)), seq_len // (down_sampling_window ** i))
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

class DecomposableMultiScaleFusion(nn.Module):
    def __init__(self, configs, down_sampling_layers):
        super(DecomposableMultiScaleFusion, self).__init__()
        self.down_sampling_layers =down_sampling_layers
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule(configs.seq_len, self.down_sampling_layers, configs.down_sampling_window)

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

class Model(nn.Module):

    def __init__(self, configs, in_channels, out_channels, down_sampling_layers, T_out):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.down_pool = nn.AvgPool2d((self.configs.down_sampling_window, 1))
        self.DMSF_blocks = nn.ModuleList([DecomposableMultiScaleFusion(configs, self.down_sampling_layers) for _ in range(configs.e_layers)])

        self.layer = configs.e_layers

        self.gcn_Layer = nn.ModuleList([
                DynamicGCN(in_channels, out_channels)
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
            out = self.gcn_Layer[i](x_in)
            enc_out_list.append(out)

        # 跨尺度融合
        for i in range(self.layer):
            enc_out_list = self.DMSF_blocks[i](enc_out_list)

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
