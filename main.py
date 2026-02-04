import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StatusDataset, SiameseStatusDataset, BalancedBatchSampler
from networks import EnhancedEmbeddingNet, EnhancedSiameseNet
from losses import ContrastiveLoss, OnlineTripletLoss
# [修改] 导入新的指标
from metrics import AccumulatedAccuracyMetric, AverageNonzeroTripletsMetric, BatchAccuracyMetric
from trainer import fit
import argparse
import os

try:
    from evaluator import evaluate_pair
except ImportError:
    evaluate_pair = None


def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='状态标签检测 - Online Hard Mining')
    parser.add_argument('--dataset-path', type=str, default=os.path.join(ROOT_DIR, 'datasets'), help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--margin', type=float, default=1.0, help='损失函数边距')
    parser.add_argument('--model-type', type=str, choices=['siamese', 'triplet'], default='triplet', help='模型类型')
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='是否使用GPU')
    parser.add_argument('--max-samples-per-class', type=int, default=float('inf'), help='每类最大样本数')
    parser.add_argument('--save-model', type=str, default=os.path.join(ROOT_DIR, 'saved_models', 'model_final.pth'),
                        help='模型保存路径')
    parser.add_argument('--resume', type=str, default=None, help='继续训练的模型路径')
    parser.add_argument('--test-only', action='store_true', help='跳过训练，直接测试')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)

    # ==========================================
    # 定义成对的测试组
    # ==========================================
    TEST_PAIRS = [
        ('isolate_close', 'isolate_open'), # 刀闸组
        ('close', 'open')                  # 英文开关组
    ]

    # 展平列表，用于训练时排除这些所有类别
    ALL_TEST_CLASSES = [cls for pair in TEST_PAIRS for cls in pair]

    # 预处理：训练时加入强增强
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 模式 1: 训练模式 (Training Mode)
    # ==========================================
    if not args.test_only:
        print(f"=== 进入训练模式 (使用 Online Hard Mining) ===")

        # 1. 特征网络
        embedding_net = EnhancedEmbeddingNet()

        # 2. 基础数据集 (正常划分训练/验证集)
        train_dataset = StatusDataset(args.dataset_path, transform=transform_train, mode='train',
                                      exclude_classes=ALL_TEST_CLASSES)
        val_dataset = StatusDataset(args.dataset_path, transform=transform_test, mode='val',
                                    exclude_classes=ALL_TEST_CLASSES)

        if args.model_type == 'siamese':
            dataset = SiameseStatusDataset(train_dataset)
            val_ds = SiameseStatusDataset(val_dataset)
            model = EnhancedSiameseNet(embedding_net)
            loss_fn = ContrastiveLoss(args.margin)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        elif args.model_type == 'triplet':
            # 1. 模型
            model = embedding_net

            # 2. 采样器：BalancedBatchSampler
            train_batch_sampler = BalancedBatchSampler(
                train_dataset.labels, n_classes=8, n_samples=4
            )

            train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            # 3. 损失
            loss_fn = OnlineTripletLoss(args.margin)

        if args.cuda:
            model = model.cuda()

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            print(f"正在加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

        # [新增] 定义要监控的指标，加入 Batch Accuracy
        metrics = [BatchAccuracyMetric()]

        try:
            # [修改] 传入 metrics
            fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler,
                args.epochs, args.cuda, log_interval=10, metrics=metrics, start_epoch=start_epoch)
        except KeyboardInterrupt:
            print("\n检测到手动停止！正在保存当前模型...")

        print(f"正在保存最终模型至: {args.save_model}")
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'embedding_net_state_dict': embedding_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_model)

        print("训练完成。")

    # ==========================================
    # 模式 2: 测试模式
    # ==========================================
    else:
        print(f"=== 进入测试模式 ===")
        if evaluate_pair is None:
            print("无法测试：缺少 evaluator.py 中的 evaluate_pair 函数。")
            return

        if not os.path.exists(args.save_model):
            print(f"模型文件未找到: {args.save_model}")
            return

        # 循环遍历每一组进行独立测试
        for class_a, class_b in TEST_PAIRS:
            current_pair_list = [class_a, class_b]

            # 创建仅包含这一对类别的数据集
            pair_dataset = StatusDataset(args.dataset_path, transform=transform_test, mode='all',
                                         include_only_classes=current_pair_list)

            if len(pair_dataset) > 0:
                # 调用通用测试函数
                evaluate_pair(args.save_model, pair_dataset, class_a, class_b, args.cuda)
            else:
                print(f"警告: 数据集中未找到类别 {class_a} 或 {class_b} 的图片，跳过此组。")


if __name__ == '__main__':
    main()