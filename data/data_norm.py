import os
import pandas as pd
import numpy as np

def read_state_data(state_folder):
    csv_files = [os.path.join(state_folder, file) for file in os.listdir(state_folder) if file.endswith('.csv')]
    all_station_data = []
    ghi_data = []
    station_ids = []

    for csv_file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        # 假设GHI列的列名是 'GHI'
        ghi = df['GHI'].values
        # 选择除时间属性外的所有列，包括GHI列
        other_data = df.iloc[:, 5:].values  # 从第六列开始的所有列
        all_station_data.append(other_data)
        ghi_data.append(ghi)
        station_id = os.path.splitext(os.path.basename(csv_file))[0]
        station_ids.append(station_id)

    return all_station_data, ghi_data, station_ids

def calculate_stats_for_first_three_years(all_station_data):
    stats = []
    for data in all_station_data:
        # 假设每年有相同数量的时间步长，这里简单地按行数的前60%来划分前三年数据
        three_years_data = data[:int(0.7 * len(data))]
        mean = np.mean(three_years_data, axis=0)
        std = np.std(three_years_data, axis=0)
        stats.append({'mean': mean, 'std': std})
    return stats

def zero_mean_normalize_with_precomputed(data, mean, std):
    return (data - mean) / std

def process_state(state_folder, output_folder):
    # 读取特定州的数据
    all_station_data, ghi_data, station_ids = read_state_data(state_folder)

    # 计算每个站点前三年数据的均值和标准差（针对所有列）
    stats = calculate_stats_for_first_three_years(all_station_data)

    # 提取GHI列的均值和标准差
    ghi_means = np.array([stat['mean'][4] for stat in stats])   # GHI列
    ghi_stds = np.array([stat['std'][4] for stat in stats])     # GHI列

    # 将GHI列的均值和标准差拼接成一个 [2, N] 的数组
    scaler_array = np.stack((ghi_means, ghi_stds))

    # 将GHI列的均值和标准差保存为.npy文件
    state_name = os.path.basename(state_folder)
    scaler_file = os.path.join(output_folder, state_name, f"{state_name}_scaler.npy")
    os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
    np.save(scaler_file, scaler_array)

    print(f"Scaler saved to {scaler_file}")

    # 对每个站点的数据进行零均值归一化（对所有列进行归一化）
    normalized_data_list = []
    for data, stat in zip(all_station_data, stats):
        mean = stat['mean']
        std = stat['std']
        # 归一化所有列
        normalized_data = zero_mean_normalize_with_precomputed(data, mean, std)
        normalized_data_list.append(normalized_data)

    # 将归一化后的数据拼接成维度为[T, N, F]的数组
    T = len(normalized_data_list[0])
    N = len(normalized_data_list)
    F = normalized_data_list[0].shape[1]  # 特征数量

    final_array = np.zeros((T, N, F))
    for i, station_data in enumerate(normalized_data_list):
        final_array[:, i, :] = station_data

    # 保存归一化后的数据为.npy文件
    normalized_data_file = os.path.join(output_folder, state_name, f"{state_name}_norm.npy")
    os.makedirs(os.path.dirname(normalized_data_file), exist_ok=True)
    np.save(normalized_data_file, final_array)

    print(f"Normalized data saved to {normalized_data_file}")

    return stats

if __name__ == "__main__":
    ghi_data_path = './GHI_data_all_features'
    output_folder = './GHI_norm_data_all_features'

    # 获取所有州文件夹路径
    state_folders = [os.path.join(ghi_data_path, folder) for folder in os.listdir(ghi_data_path) if
                     os.path.isdir(os.path.join(ghi_data_path, folder))]

    # 处理每个州的数据并保存统计信息
    state_scalers = {}
    for state_folder in state_folders:
        state_name = os.path.basename(state_folder)
        state_scalers[state_name] = process_state(state_folder, output_folder)
