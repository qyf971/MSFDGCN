import os
import numpy as np

# 参数设定
input_dir = "./GHI_norm_data_all_features"
output_dir = "./GHI_train_val_test_data_recent_all_features"
history_len = 96
pred_lens = [96]
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

os.makedirs(output_dir, exist_ok=True)

# 遍历每个区域子文件夹
# for region in os.listdir(input_dir):
for region in ['California_cities', 'Texas_cities', 'Illinois_cities']:
    region_path = os.path.join(input_dir, region)
    if not os.path.isdir(region_path):
        continue

    file_name = f"{region}_norm.npy"
    file_path = os.path.join(region_path, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        continue

    print(f"Processing: {file_path}")
    data = np.load(file_path)  # shape: [T, N, D]
    T, N, D = data.shape

    for pred_len in pred_lens:
        X, Y = [], []
        total_len = history_len + pred_len
        for i in range(T - total_len + 1):
            x = data[i:i + history_len]  # [history_len, N, D]
            y = data[i + history_len:i + total_len]  # [pred_len, N, D]
            X.append(x)
            Y.append(y)

        X = np.array(X)  # [samples, history_len, N, D]
        Y = np.array(Y)  # [samples, pred_len, N, D]

        Y = Y[ :, :, :, 4] # GHI列

        # 划分训练/验证/测试集
        num_samples = X.shape[0]
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        X_train, Y_train = X[:train_end], Y[:train_end]
        X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
        X_test, Y_test = X[val_end:], Y[val_end:]

        # 输出路径和保存
        out_file = os.path.join(
            output_dir,
            f"{region}_h{history_len}_p{pred_len}.npz"
        )
        np.savez_compressed(
            out_file,
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val,
            X_test=X_test, Y_test=Y_test
        )
        print(f"Saved: {out_file} - samples: {num_samples}")

print("✅ 数据集构建完成")
