import torch
import numpy as np


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    通用训练循环 (已加入 KNN 真实验证)
    """
    # 恢复 Scheduler 状态
    for _ in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        # 1. 训练阶段
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        # 打印训练结果
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        # 2. 原始验证阶段 (计算 Loss)
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Val Loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        # 注意：这里我们不再看 Batch Acc，因为它在 Island 数据上是虚高的

        # 3. [新增] 真实 KNN 准确率测试
        # 使用训练集作为“参考库”，验证集作为“考题”
        print(message)  # 先打印之前的日志

        print(f"Epoch: {epoch + 1}/{n_epochs}. 正在进行真实的 KNN 泛化测试 (Train vs Val)...")
        real_acc = fast_knn_eval(model, train_loader, val_loader, cuda, k=1, sample_limit=2000)
        print(f"Epoch: {epoch + 1}/{n_epochs}. ★ Real Validation Accuracy (KNN): {real_acc:.2f}%")
        print("-" * 60)

        # 4. 更新学习率
        scheduler.step()


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    # 这个函数现在主要用于计算 Validation Loss
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def fast_knn_eval(model, train_loader, val_loader, cuda, k=1, sample_limit=2000):
    """
    [新增] 快速 KNN 评估
    原理：从训练集中抽取一部分作为“参考库”(Gallery)，用验证集作为“查询”(Query)。
    这样能测试模型是否能把“怪图(Val)”和“普通图(Train)”匹配到一起。

    sample_limit: 为了速度，限制参考库的大小 (例如只用 2000 张训练图)
    """
    model.eval()

    # 1. 构建参考库 (从训练集采样)
    gallery_embeddings = []
    gallery_labels = []

    with torch.no_grad():
        count = 0
        for data, target in train_loader:
            if cuda:
                data = data.cuda()
            emb = model.get_embedding(data)
            gallery_embeddings.append(emb.cpu())
            gallery_labels.append(target)

            count += len(data)
            if count >= sample_limit:
                break

    if len(gallery_embeddings) == 0:
        return 0.0

    gallery_embeddings = torch.cat(gallery_embeddings)
    gallery_labels = torch.cat(gallery_labels)

    # 2. 构建查询集 (所有验证集)
    query_embeddings = []
    query_labels = []

    with torch.no_grad():
        for data, target in val_loader:
            if cuda:
                data = data.cuda()
            emb = model.get_embedding(data)
            query_embeddings.append(emb.cpu())
            query_labels.append(target)

    if len(query_embeddings) == 0:
        return 0.0

    query_embeddings = torch.cat(query_embeddings)
    query_labels = torch.cat(query_labels)

    # 3. 计算 KNN 准确率
    # 注意：为了防止内存爆炸，我们分批计算距离
    correct = 0
    total = len(query_labels)

    # 将 gallery 放到 GPU 上加速计算 (如果显存够)
    if cuda:
        gallery_embeddings = gallery_embeddings.cuda()

    batch_size = 100  # 查询批次

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        q_batch = query_embeddings[i:end]
        q_label_batch = query_labels[i:end]

        if cuda:
            q_batch = q_batch.cuda()

        # 计算距离矩阵: (Batch, Gallery_Size)
        # dist = (q - g)^2 = q^2 + g^2 - 2qg
        dists = torch.cdist(q_batch, gallery_embeddings)

        # 找到最近的 k 个 (这里 k=1)
        # values, indices
        _, indices = dists.topk(k, dim=1, largest=False)

        # 获取预测标签
        # indices shape: [Batch, k]
        pred_labels = gallery_labels[indices[:, 0].cpu()]

        correct += (pred_labels == q_label_batch).sum().item()

    accuracy = 100. * correct / total
    return accuracy