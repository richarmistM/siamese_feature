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

