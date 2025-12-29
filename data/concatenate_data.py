# 该段代码对年份列表中的数据进行拼接，并只保留了每天6:00到18:00的数据。
import os
import pandas as pd

# 定义源目录和目标目录
source_dir = './GHI_data'
target_dir = './GHI_data_all_features'

# 数据参考 Multi-site solar irradiance forecasting based on adaptive spatiotemporal graph convolutional network

# 指定要保留的年份列表（以字符串形式存储年份前缀）
selected_years = ['2019', '2020']  # 可根据需要修改

# 如果目标目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

states = ['California_cities', 'Texas_cities', 'Illinois_cities']
# states = ['Illinois_cities']

# 遍历州文件夹下的站点ID文件夹
# for state in os.listdir(source_dir):
for state in states:
    state_path = os.path.join(source_dir, state)
    if os.path.isdir(state_path):
        # 创建对应州的文件夹
        target_state_dir = os.path.join(target_dir, state)
        if not os.path.exists(target_state_dir):
            os.makedirs(target_state_dir)

        for site_id in os.listdir(state_path):
            site_id_path = os.path.join(state_path, site_id)
            if os.path.isdir(site_id_path):
                # 获取该站点ID下所有指定年份的csv文件路径，并按年份排序
                csv_files = [
                    os.path.join(site_id_path, f)
                    for f in os.listdir(site_id_path)
                    if f.endswith('.csv') and f[:4] in selected_years
                ]
                csv_files.sort()

                # columns_to_exclude = {7, 11, 13}
                # columns_to_use = [i for i in range(15) if i not in columns_to_exclude]

                # print(f"将保留的原始列索引: {columns_to_use}")

                # 读取并拼接数据，同时筛选每天6:00到18:00的数据
                data_frames = []
                for file in csv_files:
                    # df = pd.read_csv(file, skiprows=2, header=0, usecols=columns_to_use)
                    df = pd.read_csv(file, skiprows=2, header=0, usecols=range(15))
                    # 提取前五列并重命名
                    df.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute'] + list(df.columns[5:])
                    # 截取每天数据
                    filtered_df = df[(df['Hour'] >= 4) & (df['Hour'] <= 19)]
                    data_frames.append(filtered_df)

                # 拼接后的DataFrame
                combined_df = pd.concat(data_frames, ignore_index=True)

                # 将拼接后的数据保存到目标目录下对应的州文件夹中，以站点ID命名
                output_file = os.path.join(target_state_dir, f'{site_id}.csv')
                combined_df.to_csv(output_file, index=False)

print("所有站点ID的数据已根据指定年份筛选并拼接完成，保存至GHI_data_all文件夹中相应的州目录下。")





