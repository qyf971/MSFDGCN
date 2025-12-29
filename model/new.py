import torch
import torch.nn as nn
from GNN_Layer.DynamicGraphConv import DynamicGraphConv


class DynamicGCN(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(DynamicGCN, self).__init__()
        self.gcn_module = DynamicGraphConv(in_channels, hidden_size, hidden_size)

    def forward(self, x):
        """
        :param x: [B, N, T_in, D]
        :return: [B, N, T_out, hidden_size]
        """
        _, _, t, _ = x.size()
        out = []
        for i in range(t):
            out.append(self.gcn_module(x[:, :, i, :]))  # [B, N, hidden_size]
        out = torch.stack(out, dim=2)  # [B, N, T_in, hidden_size]
        return out

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
    def __init__(self, configs):
        super(DecomposableMultiScaleFusion, self).__init__()
        # 设置分解方式
        if configs.decomposition_method == 'moving_avg':
            self.decomposition = series_decomposition_2d(configs.moving_avg)
        elif configs.decomposition_method == "dft_decomposition":
            self.decomposition = DFT_series_decomposition(configs.top_k)
        else:
            raise ValueError('decomposition is error')

        # 多尺度融合模块
        self.multi_scale_fusion = MultiScaleFusionModule(configs.seq_len, configs.down_sampling_layers, configs.down_sampling_window)

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
    def __init__(self, T_in, T_out, hidden_size):
        super(PredictionLayer, self).__init__()

        # 缩小时间维度。
        self.conv1 = nn.Conv2d(T_in, T_out, 1)
        # 缩小通道数，降到1维。
        self.conv2 = nn.Conv2d(hidden_size, 1, 1)

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

    def __init__(self, configs):
        super(Model, self).__init__()
        self.gcn_Layer = nn.ModuleList([
                DynamicGCN(configs.in_channels, configs.hidden_size)
            for _ in range(self.down_sampling_layers + 1)
        ])
        self.down_pool = nn.AvgPool2d((configs.down_sampling_window, 1))
        self.DMSF_blocks = nn.ModuleList([DecomposableMultiScaleFusion(configs) for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        # 预测层
        self.predict_layers = torch.nn.ModuleList( # 每个尺度时间序列数据的预测层
            [
                PredictionLayer(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                    configs.hidden_size,
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
