import torch
import os
import torch.nn.functional as F
from networks import EnhancedEmbeddingNet  # 确保能导入网络结构


def load_model_from_file(model_path, cuda=True):
    """
    辅助函数：从文件加载模型权重
    """
    print(f"正在从文件加载模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 1. 初始化网络骨架
    embedding_net = EnhancedEmbeddingNet()

    if cuda:
        embedding_net = embedding_net.cuda()

    # 2. 加载权重
    device = torch.device("cuda" if cuda else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # 3. 智能匹配权重 Key
    if 'embedding_net_state_dict' in checkpoint:
        embedding_net.load_state_dict(checkpoint['embedding_net_state_dict'])
        # print("成功加载独立特征提取网络权重 (embedding_net_state_dict)")

    elif 'model_state_dict' in checkpoint:
        # print("正在从完整模型中提取特征网络权重...")
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('embedding_net.'):
                new_key = k.replace('embedding_net.', '', 1)
                new_state_dict[new_key] = v

        if len(new_state_dict) > 0:
            embedding_net.load_state_dict(new_state_dict)
        else:
            try:
                embedding_net.load_state_dict(state_dict)
            except:
                raise RuntimeError("无法匹配权重，请检查保存格式。")
    else:
        try:
            embedding_net.load_state_dict(checkpoint)
        except:
            raise RuntimeError("未知的模型文件格式。")

    embedding_net.eval()
    return embedding_net


def compute_embeddings_batched(model, dataset, indices, batch_size=32, cuda=True):
    """
    分批次计算特征向量，防止 OOM (显存溢出)
    """
    embeddings_list = []
    total = len(indices)

    for i in range(0, total, batch_size):
        batch_indices = indices[i: min(i + batch_size, total)]

        # 收集这一个 batch 的图片
        batch_imgs = []
        for idx in batch_indices:
            img, _ = dataset[idx]
            batch_imgs.append(img)

        if len(batch_imgs) == 0:
            continue

        # 堆叠成 Tensor
        batch_tensor = torch.stack(batch_imgs)

        if cuda:
            batch_tensor = batch_tensor.cuda()

        # 提取特征
        with torch.no_grad():
            emb = model.get_embedding(batch_tensor)
            # 关键：计算完立刻转回 CPU，释放 GPU 显存
            embeddings_list.append(emb.cpu())

    # 将所有批次结果拼接
    if len(embeddings_list) > 0:
        return torch.cat(embeddings_list, dim=0)
    else:
        return torch.tensor([])


def evaluate_pair(model_path, test_dataset, class_a_name, class_b_name, cuda=True):
    """
    [新增] 通用的成对测试逻辑：
    计算 class_a 和 class_b 的中心点，然后测试准确率
    """
    # 1. 加载模型
    model = load_model_from_file(model_path, cuda)

    print(f"\n--- 开始测试分组: {class_a_name} vs {class_b_name} ---")

    # 2. 获取索引 (使用传入的参数名)
    idx_a = [i for i, label in enumerate(test_dataset.labels)
             if test_dataset.classes[label] == class_a_name]
    idx_b = [i for i, label in enumerate(test_dataset.labels)
             if test_dataset.classes[label] == class_b_name]

    print(f"类别 '{class_a_name}' 样本数: {len(idx_a)}")
    print(f"类别 '{class_b_name}' 样本数: {len(idx_b)}")

    if len(idx_a) < 2 or len(idx_b) < 2:
        print(f"样本过少，无法进行测试 ({class_a_name} vs {class_b_name})。")
        return 0.0

    # 3. 划分 Support (参考) / Query (测试)
    def split_indices(indices):
        split = len(indices) // 2
        return indices[:split], indices[split:]

    ref_a_idx, query_a_idx = split_indices(idx_a)
    ref_b_idx, query_b_idx = split_indices(idx_b)

    print("正在分批提取特征...")

    # 4. 提取特征
    emb_ref_a = compute_embeddings_batched(model, test_dataset, ref_a_idx, batch_size=32, cuda=cuda)
    emb_ref_b = compute_embeddings_batched(model, test_dataset, ref_b_idx, batch_size=32, cuda=cuda)

    # 计算中心点 (Prototypes) - 都在 CPU 上进行
    prototype_a = emb_ref_a.mean(dim=0)
    prototype_b = emb_ref_b.mean(dim=0)

    # 提取测试集特征
    emb_query_a = compute_embeddings_batched(model, test_dataset, query_a_idx, batch_size=32, cuda=cuda)
    emb_query_b = compute_embeddings_batched(model, test_dataset, query_b_idx, batch_size=32, cuda=cuda)

    print("特征提取完成，开始计算距离...")

    correct = 0
    total = 0

    # --- 预测 Class A ---
    # 理论上应该离 prototype_a 更近
    d_a_to_a = torch.sum((emb_query_a - prototype_a) ** 2, dim=1)
    d_a_to_b = torch.sum((emb_query_a - prototype_b) ** 2, dim=1)

    correct += (d_a_to_a < d_a_to_b).sum().item()
    total += len(emb_query_a)

    # --- 预测 Class B ---
    # 理论上应该离 prototype_b 更近
    d_b_to_a = torch.sum((emb_query_b - prototype_a) ** 2, dim=1)
    d_b_to_b = torch.sum((emb_query_b - prototype_b) ** 2, dim=1)

    correct += (d_b_to_b < d_b_to_a).sum().item()
    total += len(emb_query_b)

    acc = 100. * correct / total
    print(f"测试完成。")
    print(f"测试集大小: {total}")
    print(f"识别准确率 ({class_a_name} vs {class_b_name}): {acc:.2f}%")
    print("---------------------------------------")

    return acc