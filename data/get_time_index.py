# import pandas as pd
#
# # 参数设置
# input_len = 96
# pred_len = 96
# train_ratio = 0.7
# val_ratio = 0.1
# test_ratio = 0.2
# file_path = './GHI_data_all/California_cities/813784.csv'
#
# # 读取数据
# df = pd.read_csv(file_path)
#
# # 构造完整的时间戳列
# df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
#
# # 保留时间列 + 特征列（按需）
# timestamps = df['Datetime'].reset_index(drop=True)
#
# # 可生成的样本总数
# total_samples = len(df) - input_len - pred_len + 1
#
# # 划分训练、验证、测试集
# train_end = int(total_samples * train_ratio)
# val_end = train_end + int(total_samples * val_ratio)
# test_start = val_end
# test_end = total_samples
#
# # 收集测试集对应的时间索引
# records = []
# for i in range(test_start, test_end):
#     x_start_time = timestamps[i]  # X 的起始时间
#     y_times = timestamps[i + input_len : i + input_len + pred_len]  # 未来6个时间点
#     record = {
#         'X_start_time': x_start_time,
#     }
#     for j in range(pred_len):
#         record[f'y_time_{j}'] = y_times.iloc[j]
#     records.append(record)
#
# # 保存结果
# output_df = pd.DataFrame(records)
# output_df.to_csv('./GHI_time_index/test_time_index_s96_p96.csv', index=False)
#
# print("测试集时间索引已生成并保存为 test_time_index.csv。")


import pandas as pd
import numpy as np
import os

# === 参数设置 ===
file_path = "./GHI_data_all/California_cities/813784.csv"
save_dir = "./GHI_time_index"
os.makedirs(save_dir, exist_ok=True)

input_len = 96
pred_len = 96
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# === 读取数据 ===
df = pd.read_csv(file_path)

# 检查前五列是否包含时间信息
time_cols = ["Year", "Month", "Day", "Hour", "Minute"]
if not all(col in df.columns[:5] for col in time_cols):
    raise ValueError("前五列应为 Year, Month, Day, Hour, Minute")

# 生成时间戳列
df["timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])

# === 确定样本起止索引 ===
total_len = len(df)
seq_len = input_len + pred_len
num_samples = total_len - seq_len + 1  # 可形成的序列样本数

# 数据集划分（样本级别，而不是时间点级别）
train_end = int(num_samples * train_ratio)
val_end = int(num_samples * (train_ratio + val_ratio))

# 测试集样本索引范围
test_start = val_end
test_end = num_samples

print(f"总样本数: {num_samples}")
print(f"训练集样本: [0, {train_end})")
print(f"验证集样本: [{train_end}, {val_end})")
print(f"测试集样本: [{test_start}, {test_end})")

# === 获取测试集对应的时间戳 ===
x_timestamps = []
y_timestamps = []

for i in range(test_start, test_end):
    x_start = i
    x_end = i + input_len
    y_start = x_end
    y_end = y_start + pred_len

    if y_end > total_len:
        break

    x_time = df["timestamp"].iloc[x_start:x_end].to_list()
    y_time = df["timestamp"].iloc[y_start:y_end].to_list()

    x_timestamps.append(x_time)
    y_timestamps.append(y_time)

# === 保存为 CSV 文件 ===
x_df = pd.DataFrame(x_timestamps)
y_df = pd.DataFrame(y_timestamps)

x_csv_path = os.path.join(save_dir, "test_x_timestamps.csv")
y_csv_path = os.path.join(save_dir, "test_y_timestamps.csv")

x_df.to_csv(x_csv_path, index=False)
y_df.to_csv(y_csv_path, index=False)

print(f"✅ 测试集时间戳已保存为：\n{x_csv_path}\n{y_csv_path}")
print(f"测试集样本数量: {len(x_timestamps)}")
print("✅ 示例：第一个测试样本的时间范围：")
print(f"输入 X: {x_timestamps[0][0]} → {x_timestamps[0][-1]}")
print(f"输出 Y: {y_timestamps[0][0]} → {y_timestamps[0][-1]}")


