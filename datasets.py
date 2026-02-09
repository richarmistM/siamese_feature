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
    [调试增强版] 支持混合结构，并会打印详细路径检查信息
    """

    def __init__(self, root_dir, transform=None, mode='train',
                 exclude_classes=None, include_only_classes=None,
                 val_islands_dict=None):

        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.val_islands_dict = val_islands_dict if val_islands_dict else {}

        # 完整类别名单
        all_classes = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'blank', '-', 'he', 'fen',
            'open', 'close',
            'IndicatorLight-Bright', 'IndicatorLight-Dark',
            'isolate_close', 'isolate_open',
            'I', 'O'
        ]

        # 过滤需要加载的类别
        if include_only_classes:
            self.classes = [c for c in all_classes if c in include_only_classes]
            # 调试：打印一下到底过滤剩下了谁
            print(f"DEBUG: 过滤模式开启，只加载: {self.classes}")
        elif exclude_classes:
            self.classes = [c for c in all_classes if c not in exclude_classes]
        else:
            self.classes = all_classes

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.img_paths = []
        self.labels = []

        print(f"--- 正在初始化数据集 ({self.mode}) 根目录: {root_dir} ---")

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)

            # --- [调试] 检查路径 ---
            if not os.path.exists(class_dir):
                # 只在测试特定类时报错，避免刷屏
                if include_only_classes:
                    print(f"❌ [跳过] 找不到文件夹: {class_dir}")
                continue

            # 获取子内容
            sub_items = os.listdir(class_dir)
            # 检查是否有 Island 文件夹
            island_dirs = [d for d in sub_items if
                           os.path.isdir(os.path.join(class_dir, d)) and d.startswith('Island_')]

            # === 分支 A: Island 结构 ===
            if len(island_dirs) > 0:
                target_val_ids = self.val_islands_dict.get(class_name, [])
                loaded_count = 0
                for item in island_dirs:
                    try:
                        island_id = int(item.split('_')[1])
                    except:
                        continue

                    is_val_island = island_id in target_val_ids

                    should_load = False
                    if self.mode == 'train' and not is_val_island:
                        should_load = True
                    elif self.mode == 'val' and is_val_island:
                        should_load = True
                    elif self.mode == 'all':
                        should_load = True

                    if should_load:
                        item_path = os.path.join(class_dir, item)
                        imgs = [f for f in os.listdir(item_path) if
                                f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
                        for img_name in imgs:
                            self.img_paths.append(os.path.join(item_path, img_name))
                            self.labels.append(self.class_to_idx[class_name])
                        loaded_count += len(imgs)
                # print(f"  -> 类 '{class_name}': Island模式加载 {loaded_count} 张")

            # === 分支 B: 普通扁平结构 (关键分支!) ===
            else:
                # 只有在 train 或 all 模式下加载，或者验证模式下(虽然通常不推荐但为了兼容)
                if self.mode in ['train', 'all']:
                    imgs = [f for f in sub_items if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]

                    if len(imgs) > 0:
                        # print(f"DEBUG: 发现普通类别 '{class_name}'，包含 {len(imgs)} 张图片 -> 加载中...")
                        for img_name in imgs:
                            self.img_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
                    else:
                        if include_only_classes:
                            print(f"⚠️ [警告] 类 '{class_name}' 文件夹存在，但没有图片！(内容: {sub_items[:5]}...)")
                else:
                    # mode='val' 但没有 Island，通常跳过，除非你需要全量验证
                    pass

        self.labels = torch.tensor(self.labels)
        print(f"[{self.mode}] 加载完成。总计: {len(self.img_paths)} 张")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx].item()
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label


# 兼容类
class SiameseStatusDataset(Dataset):
    def __init__(self, d): self.d = d

    def __getitem__(self, i): return self.d[i]

    def __len__(self): return len(self.d)


class TripletStatusDataset(Dataset):
    def __init__(self, d): self.d = d

    def __getitem__(self, i): return self.d[i]

    def __len__(self): return len(self.d)