import os

# 配置你想检查的根目录
TARGET_DIR = './strict_island_results'  # 你的数据集目录
CHECK_CLASSES = ['open', 'close', 'isolate_close', 'isolate_open']

print(f"=== 正在检查目录: {os.path.abspath(TARGET_DIR)} ===\n")

if not os.path.exists(TARGET_DIR):
    print(f"❌ 致命错误：找不到主目录 {TARGET_DIR}！")
    print("   请确认你是否在正确的目录下运行脚本，或者文件夹名字是否写错。")
    exit()

print(f"✅ 主目录存在。正在扫描子文件夹...\n")

found_count = 0
for cls in CHECK_CLASSES:
    class_path = os.path.join(TARGET_DIR, cls)

    if os.path.exists(class_path):
        # 检查里面有没有图片
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        if len(files) > 0:
            print(f"✅ [成功] 发现类别 '{cls}': 路径正确，包含 {len(files)} 张图片。")
            found_count += 1
        else:
            print(f"⚠️ [警告] 发现类别 '{cls}': 文件夹存在，但是里面是空的！(0 张图)")
            print(f"   -> 请检查是否把图片复制进去了？")
    else:
        print(f"❌ [失败] 找不到类别 '{cls}' 的文件夹！")
        print(f"   -> 代码在找这个路径: {os.path.abspath(class_path)}")
        print(f"   -> 请确认你是否把 '{cls}' 文件夹复制到了 '{TARGET_DIR}' 下面？")

print(f"\n检查结果: 找到了 {found_count} / {len(CHECK_CLASSES)} 个类别。")
if found_count == len(CHECK_CLASSES):
    print("🎉 结构完美！如果 main.py 还是报错，请检查 datasets.py 是否保存。")
else:
    print("🛠 请根据上面的红色提示修正文件夹位置。")