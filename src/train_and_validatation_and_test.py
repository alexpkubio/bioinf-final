import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import h5py




# 定义的早停机制类
class EarlyStopping:
    def __init__(self, patience=6, verbose=False, delta=0, path='checkpoint.pt', monitor='val_auc'): # Reduced patience
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.monitor = monitor

    def __call__(self, metric, model):
        # 检查metric是否为NaN，一开始进行训练时曾遇到监控指标为NaN的错误，后来通过数据纠正，传入方式纠正改正了错误但保留了代码
        if np.isnan(metric):
            print("警告: 监控指标为NaN，不更新最佳模型")
            return

        #可设置的monitor，如果是val_loss,越小越好，如果是AUC和F1，越大越好
        if self.monitor in ['val_loss']:
            score = -metric
        else:
            score = metric

        #更新和保存模型
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        #适用于有verbose时，保存模型权重
        if self.verbose:
            print(f'{self.monitor} improved ({self.best_score:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

#带标签平滑机制的BCE损失函数
class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        with torch.no_grad():
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(input, target)


# 自定义数据整理生成器
def custom_collate(batch):
    seq_inputs = []
    labels = []
    original_features = []

    #对是否有原始数据的检查
    has_original_features = len(batch[0]) == 3

    #遍历样本
    for sample in batch:
        if has_original_features:
            seq_input, original_feature, label = sample
            original_features.append(original_feature)
        else:
            seq_input, label = sample

        seq_inputs.append(seq_input)
        labels.append(label)

    # 堆叠所有张量
    seq_inputs = torch.stack(seq_inputs)
    labels = torch.stack(labels)

    #返回数据
    if has_original_features:
        original_features = torch.stack(original_features)
        return seq_inputs, original_features, labels
    else:
        return seq_inputs, None, labels

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, epochs=80):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []
    val_f1s = []

    #遍历每个epoch并初始化
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0

        #绘制进度条，方便训练进度可视化
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            # 解包批次数据
            if len(batch) == 3:  # 包含原始特征
                seq_inputs, original_features, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = original_features.to(device) if original_features is not None else None
                labels = labels.to(device)
            else:  # 不包含原始特征
                seq_inputs, _, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = None
                labels = labels.to(device)

            # 前向传播
            outputs = model(seq_inputs, original_features, labels)

            # 检查输出是否包含NaN，如果含NaN会导致统计的准确率和Loss中出现NaN，这是我们不希望看到的，但事实上在运行的过程中并没有这个警示的出现
            if torch.isnan(outputs).any():
                print(f"警告: 批次 {i} 的输出包含NaN值，跳过此批次")
                nan_batches += 1
                continue

            # 在第一次循环前添加输出值检查，帮助调试
            if i == 0 and epoch == 0:
                print(f"Output range: min={outputs.min().item()}, max={outputs.max().item()}")

            # 计算损失
            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            # 检查损失是否为NaN
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"警告: 批次 {i} 的损失为NaN或Inf，跳过此批次")
                nan_batches += 1
                continue

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 检查梯度是否包含NaN
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print(f"警告: 批次 {i} 的梯度包含NaN或Inf值，跳过参数更新")
                optimizer.zero_grad()  # 清除梯度
                nan_batches += 1
                continue

            # 更新参数
            optimizer.step()

            #统计各项指标
            running_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} [Train]")

        # 如果所有批次都是NaN，则提前结束训练
        if nan_batches == len(train_loader):
            print("错误: 所有批次都包含NaN值，训练无法继续")
            return model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s

        # 计算平均训练损失和准确率的训练指标
        train_loss = running_loss / (total if total > 0 else 1)
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段，初始化参数值
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                # 解包批次数据
                if len(batch) == 3:  # 包含原始特征
                    seq_inputs, original_features, labels = batch
                    seq_inputs = seq_inputs.to(device)
                    original_features = original_features.to(device) if original_features is not None else None
                    labels = labels.to(device)
                else:  # 不包含原始特征
                    seq_inputs, _, labels = batch
                    seq_inputs = seq_inputs.to(device)
                    original_features = None
                    labels = labels.to(device)

                # 前向传播
                outputs = model(seq_inputs, original_features, labels)

                # 跳过NaN输出
                if torch.isnan(outputs).any():
                    continue

                loss = criterion(outputs, labels)

                # 检查损失是否为NaN，跳过NaN
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue

                #统计验证指标
                running_loss += loss.item() * labels.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                #收集验证集预测结果
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        # 计算平均验证损失和准确率
        val_loss = running_loss / (total if total > 0 else 1)
        val_acc = correct / total if total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 计算验证集的AUC和F1
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, np.round(all_preds))
        except ValueError:
            val_auc = 0
            val_f1 = 0

        val_aucs.append(val_auc)
        val_f1s.append(val_f1)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印统计信息
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 早停检查
        if early_stopping.monitor == 'val_loss':
            early_stopping(val_loss, model)
        elif early_stopping.monitor == 'val_auc':
            early_stopping(val_auc, model)
        elif early_stopping.monitor == 'val_f1':
            early_stopping(val_f1, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载最佳模型
    try:
        model.load_state_dict(torch.load(early_stopping.path))
    except:
        print("警告: 无法加载最佳模型，使用当前模型")

    return model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s

# 验证模型函数
def validate_model(model, val_loader, criterion, device):
    model.eval() # 验证模式
    running_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            # 解包批次数据
            if len(batch) == 3:  # 包含原始特征
                seq_inputs, original_features, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = original_features.to(device) if original_features is not None else None
                labels = labels.to(device)
            else:  # 不包含原始特征
                seq_inputs, _, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = None
                labels = labels.to(device)

            # 前向传播
            outputs = model(seq_inputs, original_features, labels)

            # 检查输出是否包含NaN
            if torch.isnan(outputs).any():
                continue

            loss = criterion(outputs, labels)

            # 检查损失是否为NaN
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue

            running_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 计算混淆矩阵
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

            #收集验证结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

    #计算评估指标
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc = roc_auc_score(all_labels, all_preds)

    print(f'Validation Loss: {running_loss / len(val_loader):.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
    print(f'Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}')

    return accuracy, precision, recall, f1, auc

# 在测试集上评估模型
def evaluate_model_on_test_set(model, test_loader, criterion, device):
    print("\nEvaluating model on test set:")

    #调用验证函数
    accuracy, precision, recall, f1, auc = validate_model(model, test_loader, criterion, device)
    return accuracy, precision, recall, f1, auc