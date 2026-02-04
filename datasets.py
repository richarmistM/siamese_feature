import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class BalancedBatchSampler(BatchSampler):
    """
    [新增] 平衡批次采样器
    作用：确保每个 Batch 包含 N 个类别，每个类别包含 K 个样本。
    这样保证了 Batch 内部一定存在正样本对和负样本对。
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # 随机选取 n_classes 个类别
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []

            for class_ in classes:
                # 获取该类别下的样本索引
                start = self.used_label_indices_count[class_]
                end = start + self.n_samples

                # 如果样本不够了，重新洗牌
                if end > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                    start = 0
                    end = self.n_samples

                indices.extend(self.label_to_indices[class_][start:end])
                self.used_label_indices_count[class_] += self.n_samples

            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size


class StatusDataset(Dataset):
    """
    基础状态标签数据集
    """

    def __init__(self, root_dir, transform=None, max_samples_per_class=float('inf'),
                 mode='train', val_split=0.2,
                 exclude_classes=None, include_only_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.mode = mode

        all_classes = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'blank', '-', 'he', 'fen',
            'open', 'close',
            'IndicatorLight-Bright', 'IndicatorLight-Dark',
            'isolate_close', 'isolate_open',
            'I', 'O'
        ]

        # 类别过滤逻辑
        if include_only_classes:
            self.classes = [c for c in all_classes if c in include_only_classes]
        elif exclude_classes:
            self.classes = [c for c in all_classes if c not in exclude_classes]
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        if self.mode == 'train' or self.mode == 'all':
            print(f"[{mode}] 当前数据集涵盖类别: {self.classes}")

        self.img_paths = []
        self.labels = []

        rng = np.random.RandomState(42)

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                all_imgs = sorted([f for f in os.listdir(class_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                rng.shuffle(all_imgs)

                if len(all_imgs) > self.max_samples_per_class:
                    all_imgs = all_imgs[:int(self.max_samples_per_class)]

                split_idx = int(len(all_imgs) * (1 - val_split))

                if self.mode == 'train':
                    selected_imgs = all_imgs[:split_idx]
                elif self.mode == 'val':
                    selected_imgs = all_imgs[split_idx:]
                else:  # 'all'
                    selected_imgs = all_imgs

                for img_name in selected_imgs:
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        # 转换为 Tensor，方便 Sampler 使用
        self.labels = torch.tensor(self.labels)

        print(f"[{self.mode}] 加载完成: 共 {len(self.img_paths)} 张图片")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx].item()  # Tensor 转 int

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------------------------------------------
# 下面这两个类保留着，防止 main.py 引用报错，但实际上 Online Mining 不需要它们
# -------------------------------------------------------------
class SiameseStatusDataset(Dataset):
    def __init__(self, status_dataset):
        self.status_dataset = status_dataset

    def __getitem__(self, index):
        # 简单实现防报错，实际不使用
        return self.status_dataset[index]

    def __len__(self): return len(self.status_dataset)


class TripletStatusDataset(Dataset):
    def __init__(self, status_dataset):
        self.status_dataset = status_dataset

    def __getitem__(self, index):
        return self.status_dataset[index]

    def __len__(self): return len(self.status_dataset)