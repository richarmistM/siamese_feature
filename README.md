# 复杂光照条件下的特征识别与度量学习系统

## 项目简介
本项目基于深度度量学习（Deep Metric Learning），构建了一个带有在线难例挖掘的孪生/三元组网络（Siamese Triplet Network）。项目的核心目标是提取图像的高维特征，通过判断图片更接近正样本还是负样本，来解决**复杂光照条件下的状态特征识别问题**。

模型不仅能够对常规指示灯（如 `IndicatorLight-Bright`/`Dark`）、数字（`0`-`9`）和字符标签（`he`, `fen`, `open`, `close`, `I`, `O`）进行高精度分类，其最大的亮点在于**强大的泛化能力**。训练后的模型可以在完全不更新权重的情况下，直接提取未见过的全新类别（如刀闸开关的 `isolate_close` 和 `isolate_open`）的特征，并通过无监督聚类实现精准的状态分离。

## 项目目录结构
推荐的项目运行目录结构如下：

```text
project_root/
├── datasets/                   # 原始数据集存放目录 (各类别按文件夹存放)
├── saved_models/               # 训练好的模型权重保存目录
├── strict_island_results/      # auto_clustering.py 生成的清洗与划分后的中间数据集
├── auto_clustering.py          # 基于 DBSCAN 的无监督数据清洗与验证集划分
├── datasets.py                 # 数据加载器与平衡批次采样器 (BalancedBatchSampler)
├── evaluator.py                # 类别对测试与错误样本 (Bad Case) 挖掘
├── find_similar_images.py      # Top-K 相似图片检索工具
├── hash.py                     # 基于 MD5 的数据集严格去重工具
├── losses.py                   # 在线三元组损失函数 (Online Triplet Loss)
├── main.py                     # 训练与测试的主入口脚本
├── metrics.py                  # 准确率与三元组损失评估指标
├── networks.py                 # 基于 ResNet18 的特征提取骨干网络
├── test_isolate_clustering.py  # 未见类别 (刀闸) 的无监督聚类泛化测试
├── test_text_clustering.py     # 文本特征 (open/close) 的聚类测试
├── trainer.py                  # 包含 KNN 真实验证的训练循环核心逻辑
└── utils.py                    # 距离矩阵计算与样本挖掘工具
```

# 环境安装与部署

## 安装依赖包
```bash
pip install -r requirements.txt
```
> **注**：若需使用 GPU 加速，建议根据你的 CUDA 版本前往 PyTorch 官网获取对应的 `torch` 和 `torchvision` 安装命令。

---

# 运行流水线与命令

项目的完整运行分为四个阶段：**数据清洗**、**岛屿聚类划分**、**模型训练**、**泛化测试**。

## 阶段 1：数据去重 (Data Cleaning)
清理 `datasets/` 目录下的完全重复的图片，防止数据泄露影响评估真实性。
```bash
python hash.py
```

## 阶段 2：数据特征岛屿化与划分 (Data Clustering & Splitting)
使用预训练模型提取特征，并通过 t-SNE 和 DBSCAN 将外观差异巨大的同类图片划分为不同的“岛屿”，实现极具挑战性的训练/验证集分离。

1. 在 `auto_clustering.py` 中修改 `SOURCE_DIR` 指向目标类别。
2. 运行脚本并根据终端提示交互式选择验证集岛屿：
```bash
python auto_clustering.py
```

## 阶段 3：模型训练 (Model Training)
主训练脚本，默认使用 `OnlineTripletLoss` 和难例挖掘。你可以在 `main.py` 中的 `VAL_ISLAND_CONFIG` 字典里配置各类别对应的验证集岛屿 ID。

```bash
# 基础训练命令
python main.py

# 自定义参数训练
python main.py --batch-size 64 --epochs 50 --lr 0.0002 --margin 1.0 --cuda
```
> **提示**：训练完成后，模型权重将自动保存在 `saved_models/model_island.pth`。

## 阶段 4：模型评估与泛化测试 (Evaluation & Testing)

### 1. 基础成对测试与 Bad Case 导出
利用中心点计算两类之间的准确率，并输出误判图片路径：
```bash
python main.py --test-only
```

### 2. 刀闸状态无监督聚类泛化测试（核心亮点）
在不提供任何标签的情况下，测试模型对未参与训练的刀闸开闭状态（`isolate_open` vs `isolate_close`）的特征区分能力：
```bash
python test_isolate_clustering.py
```

### 3. 文本字符聚类测试
测试模型对相似文本标签（如 `open` vs `close`）的字形特征提取能力：
```bash
python test_text_clustering.py
```

### 4. 图像特征检索 (Image Retrieval)
给定一张查询图片，在指定的图库类别中寻找最相似的 Top-10 图片并可视化：

1. 打开 `find_similar_images.py`，修改 `QUERY_IMG_PATH` 为查询图片路径。
2. 运行脚本：
```bash
python find_similar_images.py
```
