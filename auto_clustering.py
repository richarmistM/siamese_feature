import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ================= 配置区域 =================
SOURCE_DIR = './datasets/1'  # 目标图片文件夹
OUTPUT_DIR = './strict_island_results/1'

# 【关键参数】扫描半径 (Epsilon)
# t-SNE 的坐标范围通常在 -100 到 100 之间。
# eps = 3.0 到 5.0 是一个经验值。
# - 如果分出来的岛屿太碎（几百个），把这个值调大 (比如 6.0)。
# - 如果岛屿还是粘连在一起，把这个值调小 (比如 3.0)。
DBSCAN_EPS = 4.5

# 【关键参数】最小样本数
# 至少有多少张图聚在一起才算一个岛？
# 设为 20 意味着孤立的几张怪图会被当成噪音扔掉。
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

    # 3. DBSCAN 密度聚类
    print(f"3. 执行 DBSCAN (eps={DBSCAN_EPS}, min_samples={MIN_SAMPLES})...")
    # 不需要指定 cluster 数量，它自动算
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(X_embedded)
    labels = db.labels_

    # 统计结果
    # label -1 代表噪音 (不属于任何岛屿)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"   -> 发现 {n_clusters_} 个独立岛屿")
    print(f"   -> 剔除 {n_noise_} 个过渡/噪音样本 (将被忽略)")

    # 4. 搬运与绘图
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"4. 生成结果图与文件夹...")

    # 绘图设置
    plt.figure(figsize=(12, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    counts = {}

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X_embedded[class_member_mask]

        if k == -1:
            # 噪音点：画成灰色小点
            col = [0.8, 0.8, 0.8, 0.5]  # 灰色
            label_name = "Noise (Ignored)"
            # 不创建文件夹，或者创建一个 ignored 文件夹
            target_folder = os.path.join(OUTPUT_DIR, "Noise_Ignored")
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='none', markersize=3, label=label_name)
        else:
            # 核心岛屿：画大一点的彩色点
            label_name = f"Island_{k}"
            target_folder = os.path.join(OUTPUT_DIR, label_name)

            # 画点
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8, label=label_name)

            # 标注中心 ID
            centroid = np.mean(xy, axis=0)
            plt.text(centroid[0], centroid[1], str(k), fontsize=18, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

        # 搬运文件
        os.makedirs(target_folder, exist_ok=True)
        current_filenames = np.array(filenames_list)[class_member_mask]

        for fname in current_filenames:
            src = os.path.join(SOURCE_DIR, fname)
            dst = os.path.join(target_folder, fname)
            shutil.copy2(src, dst)

        counts[label_name] = len(current_filenames)

    plt.title(f"Strict Isolation: {n_clusters_} Islands (Grey = Discarded Connections)")
    plt.grid(True, alpha=0.2)

    # 避免图例太多遮挡
    if n_clusters_ < 15:
        plt.legend(loc='best')

    save_path = os.path.join(OUTPUT_DIR, 'strict_islands_map.png')
    plt.savefig(save_path, dpi=300)

    print("\n=== 严格隔离完成！ ===")
    print("各组数量统计：")
    for name, count in counts.items():
        print(f"  {name}: {count} 张")

    print(f"\n【怎么看结果】")
    print(f"1. 打开图片: {save_path}")
    print("2. 【灰色】的点是‘桥梁’或‘杂质’，已经被扔进了 Noise_Ignored 文件夹。")
    print("3. 【彩色】的团块是真正的‘核心岛屿’。")
    print("4. 找一个离其他颜色最远、周围灰色区域最宽的岛屿（比如 Island_X），那个就是最完美的测试集。")


if __name__ == '__main__':
    main()