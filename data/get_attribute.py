import os
import pandas as pd

# 定义源目录和目标目录
source_dir = '../GHI_data'
attributes_dir = '../GHI_data_attributes'

# 如果目标目录不存在，则创建
if not os.path.exists(attributes_dir):
    os.makedirs(attributes_dir)

# 遍历州文件夹下的站点ID文件夹
for state in os.listdir(source_dir):
    state_path = os.path.join(source_dir, state)
    if os.path.isdir(state_path):
        # 创建对应州的文件夹
        target_state_dir = os.path.join(attributes_dir, state)
        if not os.path.exists(target_state_dir):
            os.makedirs(target_state_dir)

        # 初始化存储当前州经纬度与海拔数据的列表
        attributes_list = []

        for site_id in os.listdir(state_path):
            site_id_path = os.path.join(state_path, site_id)
            if os.path.isdir(site_id_path):
                # 获取该站点ID下所有csv文件的路径列表
                csv_files = [os.path.join(site_id_path, f) for f in os.listdir(site_id_path) if f.endswith('.csv')]
                if csv_files:
                    # 只读取第一个CSV文件的前两行以获取属性数据
                    with open(csv_files[0], 'r') as file:
                        lines = file.readlines()[:2]

                    headers = lines[0].strip().split(',')
                    data = lines[1].strip().split(',')

                    latitude_index = None
                    longitude_index = None
                    elevation_index = None

                    for i, header in enumerate(headers):
                        if 'Latitude' in header:
                            latitude_index = i
                        elif 'Longitude' in header:
                            longitude_index = i
                        elif 'Elevation' in header:
                            elevation_index = i

                    # 确保所有索引存在
                    if latitude_index is not None and longitude_index is not None:
                        latitude = float(data[latitude_index])
                        longitude = float(data[longitude_index])
                        elevation = float(data[elevation_index]) if elevation_index is not None else None

                        # 将属性数据添加到列表
                        attributes_list.append({
                            'Site_ID': site_id,
                            'Latitude': latitude,
                            'Longitude': longitude,
                            'Elevation': elevation
                        })

        # 保存为 DataFrame 并排序
        if attributes_list:
            attributes_df = pd.DataFrame(attributes_list).sort_values(by='Site_ID')

            # 保存到对应文件
            attributes_output_file = os.path.join(target_state_dir, 'Latitude_Longitude_Elevation.csv')
            attributes_df.to_csv(attributes_output_file, index=False)

print("站点ID的经纬度与海拔数据已保存到 GHI_data_attributes 文件夹下对应州的 'Latitude_Longitude_Elevation.csv' 文件中，并按 Site_ID 排序。")
