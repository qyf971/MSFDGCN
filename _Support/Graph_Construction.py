import math
import os
import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度坐标之间的大圆距离（单位：千米）。
    """
    # 地球平均半径，单位为公里
    R = 6371.004

    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine 公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


# 空间相关子图
def calculate_adjacency_matrix(state_name):
    """
    根据指定州的经纬度计算邻接矩阵 A，同时返回 edge_index 和 edge_weights。
    :param state_name: 州名
    :return: 邻接矩阵 A, edge_index 和 edge_weights
    """
    folder_path = f'./GHI_data_attributes/{state_name}'
    file_path = os.path.join(folder_path, 'Latitude_Longitude_Elevation.csv')
    print("目标经纬度文件为" + file_path)

    # 读取经纬度数据
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')
    num_sites = len(latitudes)

    # 创建一个 n×n 的距离矩阵
    distance_matrix = np.zeros((num_sites, num_sites))

    # 计算所有站点两两之间的距离
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    # 计算距离矩阵的平均值和标准差
    mean_distance = np.mean(distance_matrix.flatten())
    std_deviation = np.std(distance_matrix.flatten())

    # 创建邻接矩阵
    adj_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            if distance_matrix[i, j] <= mean_distance:
                adj_matrix[i, j] = np.exp(-(distance_matrix[i, j] ** 2) / (std_deviation ** 2))
                adj_matrix[j, i] = adj_matrix[i, j]

    # 获取边索引和边权重
    mask = adj_matrix > 0
    edge_index = np.array(np.where(mask))
    edge_weights = adj_matrix[mask]

    return adj_matrix, edge_index, edge_weights