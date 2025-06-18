import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import h5py

# 主函数
def main():
    # 获取当前脚本所在目录，加载三个集的文件夹
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'train')
    val_dir = os.path.join(script_dir, 'validation')
    test_dir = os.path.join(script_dir, 'test')

    # 创建数据集实例
    train_dataset = CpGIslandDataset(train_dir, augment=True)
    val_dataset = CpGIslandDataset(val_dir, augment=False)  # 验证集不使用增强
    test_dataset = CpGIslandDataset(test_dir, augment=False)  # 测试集不使用增强

    # 获取原始特征维度
    original_feature_dim = train_dataset.original_feature_dim

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")  # 新增测试集信息

    # 确定数据加载器的num_workers参数
    # 在Windows上使用0，在其他系统上使用4
    num_workers = 0 if os.name == 'nt' else 4

    # 创建数据加载器，使用修改后的自定义collate函数
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True,
                             collate_fn=custom_collate)  # 新增测试集加载器

    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CpGisland_CNN(
        original_feature_dim=original_feature_dim
    ).to(device)

    # 定义损失函数和优化器
    criterion = BCEWithLogitsLossWithSmoothing(smoothing=0.05)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)  # 降低初始学习率，增加L2正则化
    定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 早停机制初始化
    early_stopping = EarlyStopping(
        patience=6, # Reduced patience
        delta=1e-4,  # 多少个epoch没有改进就停止
        path=os.path.join(script_dir, 'best_model.pt'),
        monitor='val_auc'  # 可以选择 'val_loss', 'val_auc', 'val_f1'等中的一项，其实也可以用加权多项，但是这里认为一项也够
    )

    # 训练模型
    total_epochs = 80
    model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, epochs=total_epochs
    )

    # 绘制训练曲线
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(val_aucs, label='Validation AUC')
    plt.plot(val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Validation AUC and F1')

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_curves.png'))
    plt.close()

    # 在验证集上评估模型
    print("\nEvaluating model on validation set:")
    validate_model(model, val_loader, criterion, device)

    # 在测试集上评估模型
    evaluate_model_on_test_set(model, test_loader, criterion, device)

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(script_dir, 'final_model.pt'))
    print(f"Model saved to {os.path.join(script_dir, 'final_model.pt')}")

if __name__ == "__main__":
    # 确保在Windows上使用多进程时的安全
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()

