import os
import pandas as pd

# 定义源目录
source_dir = '../GHI_data_all'

# 只处理指定的州文件夹
target_states = ['California_cities', 'Texas_cities', 'Oregon_cities', 'Illinois_cities']

# 用于保存汇总结果
summary_list = []

for state in target_states:
    state_path = os.path.join(source_dir, state)
    if not os.path.isdir(state_path):
        print(f"⚠️ 未找到文件夹：{state_path}")
        continue

    print(f"正在处理州：{state}")

    # 遍历州文件夹下的所有CSV文件
    for file_name in os.listdir(state_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(state_path, file_name)

            try:
                df = pd.read_csv(file_path)

                # 检查是否包含 GHI 列（模糊匹配）
                ghi_col = None
                for col in df.columns:
                    if 'GHI' in col:
                        ghi_col = col
                        break

                if ghi_col is None:
                    print(f"⚠️ 文件 {file_name} 未找到 GHI 列，跳过。")
                    continue

                # 计算均值和最大值
                ghi_mean = round(df[ghi_col].mean(), 2)
                ghi_max = round(df[ghi_col].max(), 3)

                # 提取站点ID（去掉.csv后缀）
                site_id = os.path.splitext(file_name)[0]

                # 保存结果
                summary_list.append({
                    'State': state,
                    'Site_ID': site_id,
                    'GHI_mean': ghi_mean,
                    'GHI_max': ghi_max
                })

            except Exception as e:
                print(f"❌ 读取文件 {file_name} 时出错: {e}")

# 生成汇总DataFrame并保存
if summary_list:
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(by=['State', 'Site_ID'])
    output_file = os.path.join(source_dir, 'GHI_summary.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ 各州 GHI 均值与最大值已保存到：{output_file}")
else:
    print("❌ 未生成任何结果，请检查文件路径或 GHI 列名。")
