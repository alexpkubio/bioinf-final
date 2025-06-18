import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import h5py

class CpGisland_CNN(nn.Module):
    def __init__(self, seq_length=1200, num_channels=4, num_filters=(16, 32, 64),  kernel_sizes=(3, 5, 7), dropout_rate=0.6, original_feature_dim=0):
        super(CpGisland_CNN, self).__init__()
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.original_feature_dim = original_feature_dim
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_filters[0], kernel_size=kernel_sizes[0], padding=1),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for i in range(1, len(num_filters)):
            self.res_blocks.append(
                ResidualBlock(num_filters[i - 1], num_filters[i], kernel_sizes[i % len(kernel_sizes)])
            )

        self.attention = Selfattention(num_filters[-1])
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_feature_dim = num_filters[-1] * 2

        if original_feature_dim > 0:
            self.original_feature_fc = nn.Sequential(
                nn.Linear(original_feature_dim, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2)
            )

        fusion_input_dim = self.seq_feature_dim
        if original_feature_dim > 0:
            fusion_input_dim += 32

        self.fc = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, seq_input, original_features, label):
        x = self.initial_conv(seq_input)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.attention(x)
        max_pooled = self.global_max_pool(x).squeeze(-1)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        seq_features = torch.cat([max_pooled, avg_pooled], dim=1)
        orig_features = self.original_feature_fc(original_features)
        combined = torch.cat([seq_features, orig_features], dim=1)
        output = self.fc(combined)

        return output.squeeze()

class ResidualBlock(nn.Module):
    def __init__(self, inpt_channels, outpt_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(inpt_channels, outpt_channels, kernel_size, padding=kernel_size // 2, stride=2)
        self.bn1 = nn.BatchNorm1d(outpt_channels)
        self.conv2 = nn.Conv1d(outpt_channels, outpt_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(outpt_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(inpt_channels, outpt_channels, kernel_size=1, stride=2),
            nn.BatchNorm1d(outpt_channels)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class Selfattention(nn.Module):
    def __init__(self, in_channels):
        super(Selfattention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = 1e-12

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        batch_size, C, width = x.size()
        cal_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
        cal_key = self.key(x).view(batch_size, -1, width)
        cal_value = self.value(x).view(batch_size, -1, width)
        energy = torch.bmm(cal_query, cal_key)
        energy_scaled = energy / (cal_key.size(-2) ** 0.5 + self.eps)
        attention = F.softmax(energy_scaled, dim=-1)
        attention = torch.ones_like(energy_scaled) / energy_scaled.size(-1)
        outpt = torch.bmm(cal_value, attention.permute(0, 2, 1))
        outpt = outpt.view(batch_size, C, width)
        outpt = self.gamma * outpt + x
        return outpt

class CpGIslandDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        one_hot_path = os.path.join(root_dir, 'one_hot_encodings.h5')
        with h5py.File(one_hot_path, 'r') as hf:
            available_datasets = list(hf.keys())
            self.one_hot = hf[available_datasets[0]][:]

        if self.one_hot.dtype == np.object_:
            print("警告: one_hot数据类型为object，尝试转换为float32...")
            try:
                self.one_hot = np.array([sample.astype(np.float32) for sample in self.one_hot])
                print("成功转换为float32类型")
            except:
                raise ValueError("无法将one_hot数据转换为float32类型，请检查数据格式")

        labels_path = os.path.join(root_dir, 'labels.txt')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"找不到文件: {labels_path}")
        self.labels = np.loadtxt(labels_path, dtype=int)

        features_path = os.path.join(root_dir, 'features.csv')
        if not os.path.exists(features_path):
            print("警告: 找不到原始特征文件 features.csv，将不使用原始特征")
            self.features = None
            self.original_feature_dim = 0
        else:
            self.features = pd.read_csv(features_path).values

            nan_count = np.isnan(self.features).sum()
            inf_count = np.isinf(self.features).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"警告: 原始特征包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，将替换为0")
                self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=0.0)

            # 标准化特征
            self.features = (self.features - np.mean(self.features, axis=0)) / (np.std(self.features, axis=0) + 1e-8)
            self.original_feature_dim = self.features.shape[1]
            print(f"原始特征维度: {self.original_feature_dim}")

        self.augment = augment
        self.noise_level = 0.02

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        one_hot_sample = self.one_hot[idx]
        label = self.labels[idx]

        if self.augment:
            if np.random.random() > 0.5:
                noise = np.random.normal(0, self.noise_level, one_hot_sample.shape)
                one_hot_sample = np.clip(one_hot_sample + noise, 0, 1)
            if np.random.random() > 0.5:
                one_hot_sample = np.random.permutation(one_hot_sample.T).T
            if np.random.random() > 0.5:
                positions = np.random.choice(one_hot_sample.shape[1], size=int(one_hot_sample.shape[1] * 0.05), replace=False)
                for pos in positions:
                    one_hot_sample[:, pos] = np.eye(4)[np.random.randint(0, 4)]
        original_feature = torch.tensor(self.features[idx], dtype=torch.float32)

        one_hot_sample = torch.tensor(one_hot_sample, dtype=torch.float32).permute(1, 0)
        label = torch.tensor(label, dtype=torch.float32)

        return one_hot_sample, original_feature, label

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
        if np.isnan(metric):
            print("警告: 监控指标为NaN，不更新最佳模型")
            return

        if self.monitor in ['val_loss']:
            score = -metric
        else:
            score = metric

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
        if self.verbose:
            print(f'{self.monitor} improved ({self.best_score:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        with torch.no_grad():
            target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.loss(input, target)

def custom_collate(batch):
    seq_inputs = []
    labels = []
    original_features = []

    has_original_features = len(batch[0]) == 3

    for sample in batch:
        if has_original_features:
            seq_input, original_feature, label = sample
            original_features.append(original_feature)
        else:
            seq_input, label = sample

        seq_inputs.append(seq_input)
        labels.append(label)

    seq_inputs = torch.stack(seq_inputs)
    labels = torch.stack(labels)

    if has_original_features:
        original_features = torch.stack(original_features)
        return seq_inputs, original_features, labels
    else:
        return seq_inputs, None, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, epochs=80):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_aucs = []
    val_f1s = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in progress_bar:
            if len(batch) == 3:
                seq_inputs, original_features, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = original_features.to(device) if original_features is not None else None
                labels = labels.to(device)
            else:
                seq_inputs, _, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = None
                labels = labels.to(device)

            outputs = model(seq_inputs, original_features, labels)

            if torch.isnan(outputs).any():
                print(f"警告: 批次 {i} 的输出包含NaN值，跳过此批次")
                nan_batches += 1
                continue

            if i == 0 and epoch == 0:
                print(f"Output range: min={outputs.min().item()}, max={outputs.max().item()}")

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"警告: 批次 {i} 的损失为NaN或Inf，跳过此批次")
                nan_batches += 1
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs} [Train]")

        if nan_batches == len(train_loader):
            print("错误: 所有批次都包含NaN值，训练无法继续")
            return model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s

        train_loss = running_loss / (total if total > 0 else 1)
        train_acc = correct / total if total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if len(batch) == 3:
                    seq_inputs, original_features, labels = batch
                    seq_inputs = seq_inputs.to(device)
                    original_features = original_features.to(device) if original_features is not None else None
                    labels = labels.to(device)
                else:
                    seq_inputs, _, labels = batch
                    seq_inputs = seq_inputs.to(device)
                    original_features = None
                    labels = labels.to(device)

                outputs = model(seq_inputs, original_features, labels)

                if torch.isnan(outputs).any():
                    continue

                loss = criterion(outputs, labels)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue

                running_loss += loss.item() * labels.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        val_loss = running_loss / (total if total > 0 else 1)
        val_acc = correct / total if total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, np.round(all_preds))
        except ValueError:
            val_auc = 0
            val_f1 = 0

        val_aucs.append(val_auc)
        val_f1s.append(val_f1)

        scheduler.step(val_loss)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if early_stopping.monitor == 'val_loss':
            early_stopping(val_loss, model)
        elif early_stopping.monitor == 'val_auc':
            early_stopping(val_auc, model)
        elif early_stopping.monitor == 'val_f1':
            early_stopping(val_f1, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    try:
        model.load_state_dict(torch.load(early_stopping.path))
    except:
        print("警告: 无法加载最佳模型，使用当前模型")

    return model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s

def validate_model(model, val_loader, criterion, device):
    model.eval()
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
            if len(batch) == 3:
                seq_inputs, original_features, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = original_features.to(device) if original_features is not None else None
                labels = labels.to(device)
            else:
                seq_inputs, _, labels = batch
                seq_inputs = seq_inputs.to(device)
                original_features = None
                labels = labels.to(device)

            outputs = model(seq_inputs, original_features, labels)

            if torch.isnan(outputs).any():
                continue

            loss = criterion(outputs, labels)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue

            running_loss += loss.item() * labels.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # 使用sigmoid将logits转换为概率
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())

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

def evaluate_model_on_test_set(model, test_loader, criterion, device):
    print("\nEvaluating model on test set:")
    accuracy, precision, recall, f1, auc = validate_model(model, test_loader, criterion, device)
    return accuracy, precision, recall, f1, auc

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, 'train')
    val_dir = os.path.join(script_dir, 'validation')
    test_dir = os.path.join(script_dir, 'test')

    train_dataset = CpGIslandDataset(train_dir, augment=True)
    val_dataset = CpGIslandDataset(val_dir, augment=False)
    test_dataset = CpGIslandDataset(test_dir, augment=False)

    original_feature_dim = train_dataset.original_feature_dim

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True,
                             collate_fn=custom_collate)  # 新增测试集加载器

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CpGisland_CNN(
        original_feature_dim=original_feature_dim
    ).to(device)

    criterion = BCEWithLogitsLossWithSmoothing(smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        patience=6,
        delta=1e-4,
        path=os.path.join(script_dir, 'best_model.pt'),
        monitor='val_auc'  # 可以选择 'val_loss', 'val_auc', 'val_f1'
    )

    total_epochs = 80
    model, train_losses, val_losses, train_accs, val_accs, val_aucs, val_f1s = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, epochs=total_epochs
    )

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

    print("\nEvaluating model on validation set:")
    validate_model(model, val_loader, criterion, device)

    evaluate_model_on_test_set(model, test_loader, criterion, device)

    torch.save(model.state_dict(), os.path.join(script_dir, 'final_model.pt'))
    print(f"Model saved to {os.path.join(script_dir, 'final_model.pt')}")

if __name__ == "__main__":
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()

