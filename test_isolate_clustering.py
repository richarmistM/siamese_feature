import torch
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from networks import EnhancedEmbeddingNet
from datasets import StatusDataset

# ================= 配置 =================
MODEL_PATH = './saved_models/model_island.pth'  # 你的模型路径
DATA_PATH = './strict_island_results'  # 数据集路径
BATCH_SIZE = 32

# 指定要测试的“死对头”类别
CLASS_OPEN = 'isolate_open'
CLASS_CLOSE = 'isolate_close'


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 刀闸状态无监督聚类测试 (Isolate Open vs Close) ===")
    print(f"目的: 验证模型是否能自动将混合在一起的刀闸图片区分开闭状态")

    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(" 错误: 找不到模型文件！")
        return

    model = EnhancedEmbeddingNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 兼容各种保存格式
    if 'embedding_net_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['embedding_net_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()

    # 2. 准备数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 只加载这两个类
    target_classes = [CLASS_OPEN, CLASS_CLOSE]
    print(f"正在加载数据: {target_classes}")

    dataset = StatusDataset(DATA_PATH, transform=transform, mode='all', include_only_classes=target_classes)

    if len(dataset) == 0:
        print(" 错误: 未找到数据，请检查路径。")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 提取特征
    print("正在提取特征...")
    embeddings = []
    true_labels = []  # 0代表Close, 1代表Open (根据加载顺序可能不同，下面会自动匹配)

    # 建立映射表: 类名 -> 0或1
    # 我们规定: CLASS_CLOSE -> 0, CLASS_OPEN -> 1
    label_map = {CLASS_CLOSE: 0, CLASS_OPEN: 1}

    total_count = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            feats = model.get_embedding(imgs)
            embeddings.append(feats.cpu().numpy())

            # 获取真实标签对应的类名，再转为 0/1
            for label_idx in labels.numpy():
                class_name = dataset.classes[label_idx]
                if class_name in label_map:
                    true_labels.append(label_map[class_name])
                    total_count += 1

    embeddings = np.vstack(embeddings)
    true_labels = np.array(true_labels)

    print(f"\n特征提取完成。样本统计:")
    print(f"  -> {CLASS_CLOSE} (Label 0): {np.sum(true_labels == 0)} 张")
    print(f"  -> {CLASS_OPEN}  (Label 1): {np.sum(true_labels == 1)} 张")

    # 4. K-Means 盲分 (聚成2类)
    print("\n正在进行 K-Means 聚类...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    pred_clusters = kmeans.fit_predict(embeddings)

    # 5. 计算准确率 (解决聚类标签随机性问题)
    # 假设 Cluster 0 是 Close
    acc_forward = accuracy_score(true_labels, pred_clusters)
    # 假设 Cluster 0 是 Open (反转)
    acc_backward = accuracy_score(true_labels, 1 - pred_clusters)

    final_acc = max(acc_forward, acc_backward)
    best_preds = pred_clusters if acc_forward > acc_backward else (1 - pred_clusters)

    # 6. 输出结果
    print("\n" + "=" * 50)
    print(f"【测试结果】 刀闸状态分离准确率: {final_acc * 100:.2f}%")
    print("=" * 50)

    # 混淆矩阵
    cm = confusion_matrix(true_labels, best_preds)
    print("\n详细混淆矩阵:")
    print(f"                 预测:闭合(Close)   预测:断开(Open)")
    print(f"真实:闭合({CLASS_CLOSE})    {cm[0][0]}              {cm[0][1]}")
    print(f"真实:断开({CLASS_OPEN})     {cm[1][0]}              {cm[1][1]}")

    print("-" * 50)
    if final_acc > 0.95:
        print(" 评价: 极好！模型完全抓住了刀闸开闭的物理特征差异。")
    elif final_acc > 0.85:
        print(" 评价: 良好。大部分能分清，可能在某些模糊角度有混淆。")
    else:
        print(" 评价: 一般。模型对刀闸开闭的特征界限还不够清晰。")


if __name__ == '__main__':
    main()