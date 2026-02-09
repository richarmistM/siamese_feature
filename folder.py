import os

# 定义要创建的所有文件夹名称列表
folder_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'blank', '-', 'he', 'fen',
    'open', 'close',
    'IndicatorLight-Bright', 'IndicatorLight-Dark',
    'isolate_close', 'isolate_open',
    'I', 'O'
]

# ******** 请修改这里为你要创建文件夹的目标目录 ********
target_dir = r"C:\Users\27114\CLionProjects\siamese_model2\datasets\val"  # Windows示例路径（r表示原始字符串，避免转义）
# target_dir = "/home/yourname/target_dir"  # Linux/Mac示例路径

# 遍历列表，逐个创建文件夹
for folder in folder_names:
    # 拼接完整的文件夹路径（自动适配系统路径分隔符）
    folder_path = os.path.join(target_dir, folder)
    try:
        # 创建文件夹，exist_ok=True 表示如果文件夹已存在则不报错
        os.makedirs(folder_path, exist_ok=True)
        print(f"成功创建文件夹: {folder_path}")
    except Exception as e:
        print(f"创建文件夹 {folder_path} 失败，错误信息: {e}")

print("所有文件夹创建操作完成！")