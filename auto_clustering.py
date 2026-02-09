import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你想分类的那个类别的文件夹路径 (比如全是减号的文件夹)
SOURCE_DIR = './datasets/0'

# 2. 结果保存路径
OUTPUT_DIR = './clustered_results/0'

# 3. 你觉得大概有几种样式？(比如红、白、黑就是3种，如果不确定可以设大一点，比如5)
N_CLUSTERS = 3

# 4. 批处理大小
BATCH_SIZE = 32


# ===========================================

class SimpleImageDataset(Dataset):
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
    print(f"正在使用设备: {device}")

    # 1. 准备模型 (使用标准 ResNet18 提取特征)
    # 我们不需要训练它，只需要它“看”图片的能力
    print("加载预训练模型...")
    model = models.resnet18(pretrained=True)
    # 去掉最后的全连接层，直接输出 512 维特征
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # 2. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SimpleImageDataset(SOURCE_DIR, transform=transform)
    if len(dataset) == 0:
        print("错误：源文件夹里没有图片！")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"开始提取特征，共 {len(dataset)} 张图片...")

    all_features = []
    all_filenames = []

    # 3. 提取特征
    with torch.no_grad():
        for imgs, filenames in tqdm(dataloader):
            imgs = imgs.to(device)
            # 提取特征 [Batch, 512]
            features = model(imgs)
            # 转为 numpy
            features = features.cpu().numpy()

            all_features.append(features)
            all_filenames.extend(filenames)

    all_features = np.vstack(all_features)

    # 4. K-Means 聚类
    print(f"正在进行 K-Means 聚类 (K={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_features)

    # 5. 移动文件
    print("正在根据聚类结果整理文件...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 统计每个簇的数量
    counts = {i: 0 for i in range(N_CLUSTERS)}

    for filename, label in zip(all_filenames, labels):
        # 创建子文件夹
        cluster_folder = os.path.join(OUTPUT_DIR, f"style_{label}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(cluster_folder, filename)

        shutil.copy2(src_path, dst_path)
        counts[label] += 1

    print("\n分类完成！结果如下：")
    for i in range(N_CLUSTERS):
        print(f"  -> 样式 {i} (style_{i}): {counts[i]} 张图片")

    print(f"\n请去 {OUTPUT_DIR} 查看结果，并根据图片内容手动重命名文件夹。")


if __name__ == '__main__':
    main()