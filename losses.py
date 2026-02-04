import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    对比损失 (保持不变，供 Siamese 模式使用)
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    [新增] 在线难例挖掘三元组损失
    不需要外部 Selector，直接在 forward 中完成矩阵计算和难例筛选。
    """

    def __init__(self, margin=1.0):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        :param embeddings: [Batch_Size, Feature_Dim]
        :param labels: [Batch_Size]
        """
        # 1. 计算成对距离矩阵 (Pairwise Distance Matrix)
        # 避免 sqrt(0) 导致梯度 Nan，使用平方欧氏距离计算
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # 处理数值不稳定性 (保证非负)
        distances = torch.clamp(distances, min=0.0)

        # 开根号得到欧氏距离
        euclidean_dist = torch.sqrt(distances + 1e-8)

        # 2. 构建掩码 (Masks)
        labels = labels.unsqueeze(1)
        # 正样本掩码：类别相同，且不是自己 (对角线为0)
        mask_pos = (labels == labels.t()).float()
        mask_pos -= torch.eye(labels.size(0), device=labels.device)

        # 负样本掩码：类别不同
        mask_neg = (labels != labels.t()).float()

        # 3. 难例挖掘 (Hard Mining)
        # Hardest Positive: 同类中距离最远的
        # 我们把非正样本对的距离设为 0，然后取 max
        hardest_pos_dist = (euclidean_dist * mask_pos).max(1)[0]

        # Hardest Negative: 异类中距离最近的
        # 我们把非负样本对的距离设为无穷大，然后取 min
        max_dist = euclidean_dist.max()
        hardest_neg_dist = (euclidean_dist + max_dist * (1.0 - mask_neg)).min(1)[0]

        # 4. 计算损失
        # Loss = max(0, pos_dist - neg_dist + margin)
        # 这里实际上是在每个 anchor 对应的所有三元组中，选出了 loss 最大的那个
        triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)

        # 计算平均损失
        triplet_loss = triplet_loss.mean()

        return triplet_loss


# 为了兼容旧代码引用的 TripletLoss，可以保留一个空壳或者直接别名
# 如果 main.py 里不再用 TripletLoss，这里可以不写，或者保留旧的实现
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = (anchor - positive).pow(2).sum(1)
        dist_neg = (anchor - negative).pow(2).sum(1)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()