import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# ================= 配置区域 =================
SOURCE_DIR = './datasets/IndicatorLight-Bright'  # 目标图片文件夹
OUTPUT_DIR = './strict_island_results/IndicatorLight-Bright'  # 中间结果保存位置

# 【关键参数】DBSCAN 设置
DBSCAN_EPS = 4.5
MIN_SAMPLES = 20


# ===========================================

class FeatureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, img_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 提取特征... (Device: {device})")

    # 1. 特征提取
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FeatureDataset(SOURCE_DIR, transform=transform)
    if len(dataset) == 0:
        print("错误：源文件夹为空")
        return

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    features_list = []
    filenames_list = []

    with torch.no_grad():
        for imgs, names in tqdm(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            features_list.append(feats)
            filenames_list.extend(names)

    X = np.vstack(features_list)

    # 2. t-SNE 降维
    print("2. t-SNE 降维中...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # 3. DBSCAN 聚类
    print(f"3. 执行 DBSCAN (eps={DBSCAN_EPS}, min_samples={MIN_SAMPLES})...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(X_embedded)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   -> 发现 {n_clusters_} 个独立岛屿")

    # 4. 生成中间结果文件夹 (Staging Area)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"4. 生成分布图与预览文件夹...")

    # 绘图
    plt.figure(figsize=(12, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    valid_islands = []  # 记录有效的岛屿ID

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X_embedded[class_member_mask]

        if k == -1:
            # 噪音
            col = [0.8, 0.8, 0.8, 0.5]
            label_name = "Noise_Ignored"
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='none', markersize=3, label='Noise')
            target_folder = os.path.join(OUTPUT_DIR, label_name)
        else:
            # 岛屿
            label_name = f"Island_{k}"
            valid_islands.append(k)
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8, label=label_name)

            # 标注中心
            centroid = np.mean(xy, axis=0)
            plt.text(centroid[0], centroid[1], str(k), fontsize=18, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

            target_folder = os.path.join(OUTPUT_DIR, label_name)

        # 移动文件到临时文件夹
        os.makedirs(target_folder, exist_ok=True)
        current_filenames = np.array(filenames_list)[class_member_mask]
        for fname in current_filenames:
            src = os.path.join(SOURCE_DIR, fname)
            dst = os.path.join(target_folder, fname)
            shutil.copy2(src, dst)

    plt.title(f"Strict Isolation: {n_clusters_} Islands")
    plt.grid(True, alpha=0.2)
    save_path = os.path.join(OUTPUT_DIR, 'map_preview.png')
    plt.savefig(save_path, dpi=300)

    print("\n" + "=" * 50)
    print(f"中间分类完成！请查看图片: {save_path}")
    print("=" * 50)
    print(f"检测到的有效岛屿 ID: {sorted(valid_islands)}")
    print("现在你需要决定哪些岛屿做【验证集】，剩下的将自动归并为【训练集】。")
    print("注意：灰色噪音点将被自动丢弃。")

    # ==========================================
    # 5. 交互式合并逻辑 (新增部分)
    # ==========================================
    while True:
        user_input = input("\n请输入要做验证集(Val)的岛屿ID (用逗号分隔，例如 2,5): ").strip()
        if not user_input:
            print("输入为空，请重新输入。")
            continue

        try:
            # 解析输入
            val_ids = [int(x.strip()) for x in user_input.replace('，', ',').split(',') if x.strip().isdigit()]

            # 检查ID是否有效
            if not all(uid in valid_islands for uid in val_ids):
                print(f"错误：输入的 ID 包含不存在的岛屿。有效 ID 为: {valid_islands}")
                continue

            if len(val_ids) == 0:
                print("错误：至少选择一个岛屿作为验证集。")
                continue

            break
        except Exception as e:
            print(f"输入格式错误: {e}")

    print(f"\n你选择了验证集岛屿: {val_ids}")
    print("正在执行最终归并...")

    # 创建最终文件夹
    final_root = os.path.join(OUTPUT_DIR, "Final_Split")
    train_dir = os.path.join(final_root, "train")  # 这里你可以根据需要改为 'train/3'
    val_dir = os.path.join(final_root, "val")  # 这里你可以根据需要改为 'val/3'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_count = 0
    val_count = 0

    # 遍历所有有效岛屿进行搬运
    for island_id in valid_islands:
        src_island_path = os.path.join(OUTPUT_DIR, f"Island_{island_id}")

        # 决定去向
        if island_id in val_ids:
            target_base = val_dir
            is_val = True
        else:
            target_base = train_dir
            is_val = False

        # 搬运该岛屿下的所有图片
        files = os.listdir(src_island_path)
        for f in files:
            src_file = os.path.join(src_island_path, f)
            dst_file = os.path.join(target_base, f)
            shutil.copy2(src_file, dst_file)

            if is_val:
                val_count += 1
            else:
                train_count += 1

    print("\n=== 归并完成！ ===")
    print(f"结果保存在: {final_root}")
    print(f"  -> 训练集 (Train): {train_count} 张 (包含岛屿 {[i for i in valid_islands if i not in val_ids]})")
    print(f"  -> 验证集 (Val)  : {val_count} 张 (包含岛屿 {val_ids})")
    print(f"  -> 噪音数据已丢弃。")


if __name__ == '__main__':
    main()