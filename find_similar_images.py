import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm

# 引入你项目中的模块
from networks import EnhancedEmbeddingNet
from datasets import StatusDataset

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH = './saved_models/model_island.pth'

# 2. 数据集根目录
DATA_ROOT = './strict_island_results'

# 3. 指定要搜索的“图库”类别 (例如只在 'isolate_open' 里找)
SEARCH_CLASS = 'isolate_close'

# 4. 指定一张查询图片 (可以是文件夹里的某一张，也可以是外部的图片)
# 请替换成你实际存在的某张图片的绝对或相对路径
QUERY_IMG_PATH = './strict_island_results\isolate_close/aa_isolate-close_close_18_23791.jpg'


# ===========================================

def load_model(device):
    print(f"Loading model from {MODEL_PATH}...")
    model = EnhancedEmbeddingNet().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # 兼容不同的保存字典键名
    if 'embedding_net_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['embedding_net_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    return model


def get_single_embedding(model, img_path, transform, device):
    """单独读取一张图片并提取特征"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Query image not found: {img_path}")

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error opening image {img_path}: {e}")

    # 增加 batch 维度: [3, 224, 224] -> [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.get_embedding(img_tensor)
        # 归一化
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding, img


def build_gallery(model, data_loader, device):
    """构建图库的特征库"""
    gallery_embeddings = []
    gallery_paths = []

    print(f"Building gallery features for class: {SEARCH_CLASS}...")
    with torch.no_grad():
        for imgs, indices in tqdm(data_loader):
            imgs = imgs.to(device)
            embeddings = model.get_embedding(imgs)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 收集时放到 CPU 防止显存溢出
            gallery_embeddings.append(embeddings.cpu())

    # 合并所有 batch
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0)

    # 获取对应的路径
    gallery_paths = data_loader.dataset.img_paths

    return gallery_embeddings, gallery_paths


def visualize_results(query_img, results):
    """
    使用 Matplotlib 绘制结果
    results: list of (score, path)
    """
    plt.figure(figsize=(15, 6))

    # 1. 显示查询图
    plt.subplot(2, 6, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    # 2. 显示 Top 10
    for i, (score, path) in enumerate(results):
        try:
            img = Image.open(path).convert('RGB')
            plt.subplot(2, 6, i + 2)  # 从第2个位置开始画
            plt.imshow(img)
            # 提取文件名显示
            fname = os.path.basename(path)
            # 截断过长的文件名
            short_name = fname[:10] + '...' if len(fname) > 10 else fname
            plt.title(f"Rank {i + 1}\nSim: {score:.4f}\n{short_name}", fontsize=8)
            plt.axis('off')
        except:
            pass

    plt.tight_layout()
    plt.show()


def main():
    if not os.path.exists(QUERY_IMG_PATH):
        print("\n" + "!" * 50)
        print(f"❌ 错误: 未找到查询图片！")
        print(f"请打开脚本 'find_similar_images.py'")
        print(f"修改第 29 行的 QUERY_IMG_PATH 变量，指向一张真实的图片。")
        print("!" * 50 + "\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载模型
    model = load_model(device)

    # 3. 计算查询图片的特征
    print(f"Extracting query feature...")
    query_emb, query_img_obj = get_single_embedding(model, QUERY_IMG_PATH, transform, device)

    # 4. 加载图库数据
    dataset = StatusDataset(DATA_ROOT, transform=transform, mode='all',
                            include_only_classes=[SEARCH_CLASS])

    if len(dataset) == 0:
        print(f"No images found in class {SEARCH_CLASS}")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 5. 构建图库特征 (返回的是 CPU 上的 Tensor)
    gallery_embs, gallery_paths = build_gallery(model, loader, device)

    # 6. 计算相似度 (矩阵乘法)
    print("Calculating similarities...")

    # === [关键修复] 将 gallery_embs 移动到与 query_emb 相同的设备上 ===
    if query_emb.device.type == 'cuda':
        gallery_embs = gallery_embs.cuda()

    similarities = torch.mm(query_emb, gallery_embs.t()).squeeze(0)  # [N]

    # 7. 排序并取 Top 10
    topk_scores, topk_indices = torch.topk(similarities, k=11, largest=True)

    results = []
    print(f"\n=== Top Similar Images for query: {os.path.basename(QUERY_IMG_PATH)} ===")

    count = 0
    for score, idx in zip(topk_scores, topk_indices):
        idx = idx.item()
        path = gallery_paths[idx]

        # 简单判断路径是否完全相同，如果是同一张图则跳过
        if os.path.abspath(path) == os.path.abspath(QUERY_IMG_PATH):
            continue

        print(f"Rank {count + 1}: Score {score:.4f} | Path: {path}")
        results.append((score.item(), path))
        count += 1
        if count >= 10:
            break

    # 8. 可视化
    visualize_results(query_img_obj, results)


if __name__ == '__main__':
    main()