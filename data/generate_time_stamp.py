import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 参数设定
input_dir = './GHI_data_all'
output_dir = './GHI_train_val_test_time_stamp'
history_len = 96
# pred_lens = [6, 12, 24, 36, 48, 96]
pred_lens = [96]
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

os.makedirs(output_dir, exist_ok=True)

def get_timestamp_array(df):
    return df.iloc[:, :5].values.astype(int)  # Year, Month, Day, Hour, Minute

def split_data(timestamps, history_len, pred_len):
    X_timestamps = []
    Y_timestamps = []

    total_len = history_len + pred_len
    for i in range(len(timestamps) - total_len + 1):
        x = timestamps[i:i + history_len]
        y = timestamps[i + history_len:i + total_len]
        X_timestamps.append(x)
        Y_timestamps.append(y)

    return np.array(X_timestamps), np.array(Y_timestamps)

def split_train_val_test(data_array):
    total = len(data_array)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        data_array[:train_end],
        data_array[train_end:val_end],
        data_array[val_end:]
    )

# 遍历每个州文件夹
# for state_name in tqdm(os.listdir(input_dir), desc="Processing states"):
for state_name in ['California_cities']:
    state_path = os.path.join(input_dir, state_name)
    if not os.path.isdir(state_path):
        print(f"{state_path} is not a directory.")
        continue

    # 只读取一个 CSV 获取时间戳
    sample_csv_path = None
    for file in os.listdir(state_path):
        if file.endswith('.csv'):
            sample_csv_path = os.path.join(state_path, file)
            break
    if sample_csv_path is None:
        continue

    df = pd.read_csv(sample_csv_path)
    timestamps = get_timestamp_array(df)

    for pred_len in pred_lens:
        try:
            X_timestamp_seq, Y_timestamp_seq = split_data(timestamps, history_len, pred_len)

            X_train, X_val, X_test = split_train_val_test(X_timestamp_seq)
            Y_train, Y_val, Y_test = split_train_val_test(Y_timestamp_seq)

            out_file = os.path.join(output_dir, f"{state_name}_h{history_len}_p{pred_len}.npz")
            np.savez_compressed(
                out_file,
                X_timestamp_train=X_train,
                X_timestamp_val=X_val,
                X_timestamp_test=X_test,
                Y_timestamp_train=Y_train,
                Y_timestamp_val=Y_val,
                Y_timestamp_test=Y_test
            )
        except Exception as e:
            print(f"❌ Error processing {state_name} with pred_len={pred_len}: {e}")

print("✅ 时间戳文件生成完成")
