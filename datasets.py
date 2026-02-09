import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms


class BalancedBatchSampler(BatchSampler):
    """
    (保持不变) 平衡批次采样器
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
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_]
                end = start + self.n_samples
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
    [升级版] Island 结构专用数据集加载器
    逻辑：
    1. 遍历每个类别的文件夹。
    2. 忽略 'Noise_Ignored'。
    3. 识别 'Island_X'。
    4. 根据 val_islands_dict 决定该 Island 属于 Train 还是 Val。
    """

    def __init__(self, root_dir, transform=None, mode='train',
                 exclude_classes=None, include_only_classes=None,
                 val_islands_dict=None):
        """
        :param val_islands_dict: 字典配置。
               例如: {'3': [2, 5], '0': [1]}
               含义: 类别 '3' 的 Island_2 和 Island_5 是验证集，其余是训练集。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.val_islands_dict = val_islands_dict if val_islands_dict else {}

        # 定义所有类别
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

        self.img_paths = []
        self.labels = []

        # 统计加载情况，用于Debug
        self.loaded_stats = {}

        print(f"--- 正在初始化数据集 ({self.mode}) ---")

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            # 获取该类别下所有的子文件夹 (Island_0, Island_1, Noise_Ignored...)
            sub_items = os.listdir(class_dir)

            # 获取该类别指定的验证集 ID 列表 (如果没有指定，则为空，意味着全都是训练集)
            target_val_ids = self.val_islands_dict.get(class_name, [])

            count_loaded = 0

            for item in sub_items:
                item_path = os.path.join(class_dir, item)

                # 1. 必须是文件夹
                if not os.path.isdir(item_path):
                    continue

                # 2. 严格排除 Noise
                if "noise" in item.lower() or "ignored" in item.lower():
                    continue

                # 3. 解析 Island ID
                # 文件夹名通常是 "Island_0", "Island_1"
                if item.startswith("Island_"):
                    try:
                        island_id = int(item.split('_')[1])
                    except:
                        continue  # 名字不规范跳过

                    # === 核心路由逻辑 ===
                    is_val_island = island_id in target_val_ids

                    should_load = False
                    if self.mode == 'train':
                        # 训练模式：加载【非】验证集的岛屿
                        if not is_val_island:
                            should_load = True
                    elif self.mode == 'val':
                        # 验证模式：只加载验证集的岛屿
                        if is_val_island:
                            should_load = True
                    elif self.mode == 'all':
                        should_load = True

                    if should_load:
                        # 读取该岛屿下的所有图片
                        imgs = [f for f in os.listdir(item_path)
                                if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]

                        for img_name in imgs:
                            self.img_paths.append(os.path.join(item_path, img_name))
                            self.labels.append(self.class_to_idx[class_name])

                        count_loaded += len(imgs)

            if count_loaded > 0:
                self.loaded_stats[class_name] = count_loaded

        self.labels = torch.tensor(self.labels)
        print(f"[{self.mode}] 加载完成。总计: {len(self.img_paths)} 张")
        # print(f"详情: {self.loaded_stats}") # 如果需要看每个类加载了多少张，取消注释

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx].item()
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label


# 为了兼容 evaluator.py 的引用，保留这两个空壳类
class SiameseStatusDataset(Dataset):
    def __init__(self, status_dataset): self.status_dataset = status_dataset

    def __getitem__(self, index): return self.status_dataset[index]

    def __len__(self): return len(self.status_dataset)


class TripletStatusDataset(Dataset):
    def __init__(self, status_dataset): self.status_dataset = status_dataset

    def __getitem__(self, index): return self.status_dataset[index]

    def __len__(self): return len(self.status_dataset)