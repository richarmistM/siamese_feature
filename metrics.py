import numpy as np
import torch


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    累计准确率指标
    通常用于分类任务，但在度量学习中可能较少直接使用，除非加上分类头。
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    """
    平均非零三元组数量指标
    意义：统计一个 Batch 中有多少个三元组产生了 Loss (Loss > 0)。
    """

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values) if self.values else 0

    def name(self):
        return 'Average nonzero triplets'


class BatchAccuracyMetric(Metric):
    """
    [新增] Batch 内部最近邻准确率 (Batch-wise 1-NN Accuracy)
    含义：在一个 Batch 中，对于每一个样本，如果距离它最近的样本（排除自身）属于同一类，则视为正确。
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        # 兼容处理：outputs 可能是 tuple
        embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        # target 可能是 tuple
        labels = target[0] if isinstance(target, (tuple, list)) else target

        # 确保数据在同一设备
        with torch.no_grad():
            # 1. 计算距离矩阵 (Batch_Size x Batch_Size)
            # dist[i][j] = ||x_i - x_j||^2
            dot_product = torch.matmul(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            # 修正数值误差导致的负数
            distances = torch.clamp(distances, min=0.0)
            distances = torch.sqrt(distances + 1e-8)

            # 2. 将对角线（自己到自己的距离）设为无穷大，防止把自己选为“最近邻”
            distances.fill_diagonal_(float('inf'))

            # 3. 找到每个样本距离最近的样本索引
            # min_indices[i] 表示第 i 张图最近邻图的索引
            min_indices = distances.argmin(dim=1)

            # 4. 比对标签
            # 预测的类别 = 最近邻样本的实际类别
            predicted_labels = labels[min_indices]

            # 统计正确数
            batch_correct = (predicted_labels == labels).sum().item()
            batch_total = labels.size(0)

            self.correct += batch_correct
            self.total += batch_total

        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        if self.total == 0:
            return 0.0
        return 100.0 * float(self.correct) / self.total

    def name(self):
        return 'Batch Acc'