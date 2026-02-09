import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import StatusDataset, BalancedBatchSampler
from networks import EnhancedEmbeddingNet
from losses import OnlineTripletLoss
from metrics import BatchAccuracyMetric
from trainer import fit
import argparse
import os

try:
    from evaluator import evaluate_pair
except ImportError:
    evaluate_pair = None


def main():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==========================================
    # 【配置区域】在此处指定你的验证集策略
    # ==========================================
    # 格式: '类别名': [岛屿ID, 岛屿ID...]
    # 这些指定的岛屿会变成验证集 (Validation Set)
    # 未指定的岛屿会自动变成训练集 (Training Set)
    # Noise_Ignored 文件夹会自动被排除

    VAL_ISLAND_CONFIG = {
        '0': [8],#红色
        '1': [5],#蓝色
        '2': [13],#数字屏
        '3': [14],#暗紫色
        '4': [4],#有噪点
        '5': [3],#特别黑
        '6': [7],#特别小
        '7': [10],#特别糊
        '8': [12],#泛光
        '9': [13],#特别诡异
        '-': [1],#泛光
        'I': [11],#
        'O': [4,6],#色调很怪，明暗相间
        'blank': [3],#
        'fen': [9],#
        'he': [3,7],#
        'IndicatorLight-Dark': [1,9],  #
        'IndicatorLight-Bright': [7,20]#
        # 其他没写的类别，如果没有 Island 结构或没指定，默认全部进入训练集(或者你可以留空)
    }

    # 默认路径修改为你的结果文件夹
    DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, 'strict_island_results')

    parser = argparse.ArgumentParser(description='状态标签检测 - Island Split Mode')
    parser.add_argument('--dataset-path', type=str, default=DEFAULT_DATA_PATH, help='数据集路径(strict_island_results)')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--margin', type=float, default=1.0, help='损失函数边距')
    parser.add_argument('--model-type', type=str, choices=['siamese', 'triplet'], default='triplet', help='模型类型')
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='是否使用GPU')
    parser.add_argument('--save-model', type=str, default=os.path.join(ROOT_DIR, 'saved_models', 'model_island.pth'),
                        help='模型保存路径')
    parser.add_argument('--resume', type=str, default=None, help='继续训练的模型路径')
    parser.add_argument('--test-only', action='store_true', help='跳过训练，直接测试')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)

    # 定义成对的测试组 (用于最后阶段的 evaluator 测试)
    TEST_PAIRS = [
        ('isolate_close', 'isolate_open'),
        ('close', 'open')
    ]
    ALL_TEST_CLASSES = [cls for pair in TEST_PAIRS for cls in pair]

    # 数据预处理
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
    # 模式 1: 训练模式
    # ==========================================
    if not args.test_only:
        print(f"=== 进入训练模式 (Island Split Strategy) ===")
        print(f"验证集配置: {VAL_ISLAND_CONFIG}")

        embedding_net = EnhancedEmbeddingNet()

        # 1. 实例化训练集
        # 传入配置，自动排除掉做验证的 Island
        train_dataset = StatusDataset(args.dataset_path, transform=transform_train, mode='train',
                                      exclude_classes=ALL_TEST_CLASSES,
                                      val_islands_dict=VAL_ISLAND_CONFIG)

        # 2. 实例化验证集
        # 传入配置，只加载指定的 Island
        val_dataset = StatusDataset(args.dataset_path, transform=transform_test, mode='val',
                                    exclude_classes=ALL_TEST_CLASSES,
                                    val_islands_dict=VAL_ISLAND_CONFIG)

        # 检查是否成功加载
        if len(train_dataset) == 0:
            print("错误：训练集为空！请检查路径或配置。")
            return
        if len(val_dataset) == 0:
            print("警告：验证集为空。这意味着所有数据都用于训练，没有未见过的测试集。")

        # 3. 构建 DataLoader
        if args.model_type == 'triplet':
            model = embedding_net

            # 使用 Balanced Sampler 保证训练稳定
            train_batch_sampler = BalancedBatchSampler(
                train_dataset.labels, n_classes=8, n_samples=8
            )
            train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

            loss_fn = OnlineTripletLoss(args.margin)
        else:
            # Siamese 模式 (如需使用)
            raise NotImplementedError("建议使用 Triplet 模式以获得最佳效果")

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

        metrics = [BatchAccuracyMetric()]

        try:
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
        if evaluate_pair is None: return

        if not os.path.exists(args.save_model):
            print(f"模型文件未找到: {args.save_model}")
            return

        for class_a, class_b in TEST_PAIRS:
            current_pair_list = [class_a, class_b]
            # 测试时 mode='all' 表示加载所有 island，因为我们是拿保存好的模型来跑分
            pair_dataset = StatusDataset(args.dataset_path, transform=transform_test, mode='all',
                                         include_only_classes=current_pair_list)

            if len(pair_dataset) > 0:
                evaluate_pair(args.save_model, pair_dataset, class_a, class_b, args.cuda)
            else:
                print(f"警告: 数据集中未找到类别 {class_a} 或 {class_b} 的图片，跳过此组。")


if __name__ == '__main__':
    main()