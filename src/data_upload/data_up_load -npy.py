import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import h5py

# 定义模型类和传入数据
class CpGIslandDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir

        # 检查并加载one_hot.npy
        one_hot_path = os.path.join(root_dir, 'one_hot.npy')
        self.one_hot = np.load(one_hot_path, allow_pickle=True)

        # 确保one_hot数据是正确的数值类型
        if self.one_hot.dtype == np.object_:
            print("警告: one_hot数据类型为object，尝试转换为float32...")
            try:
                # 尝试将每个样本转换为float32
                self.one_hot = np.array([sample.astype(np.float32) for sample in self.one_hot])
                print("成功转换为float32类型")
            except:
                raise ValueError("无法将one_hot数据转换为float32类型，请检查数据格式")

        # 检查并处理NaN和Inf值，由于没有Inf值，所以cnn模型代码中省略了Inf的情况，注意在我们的数据处理过程中，one_hot格式一般不会出现这两种值
        nan_count = np.isnan(self.one_hot).sum()
        inf_count = np.isinf(self.one_hot).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"警告: one_hot数据包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，将替换为0")
            self.one_hot = np.nan_to_num(self.one_hot, nan=0.0, posinf=1.0, neginf=0.0)

        # 检查并加载labels.txt，这是是一个纯文本，每行一个整数标签（1 表示正样本，0 表示负样本）
        labels_path = os.path.join(root_dir, 'labels.txt')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"找不到文件: {labels_path}")
        self.labels = np.loadtxt(labels_path, dtype=int)

        # 检查并加载features.csv (原始特征)
        features_path = os.path.join(root_dir, 'features.csv')
        if not os.path.exists(features_path):
            print("警告: 找不到原始特征文件 features.csv，将不使用原始特征")
            self.features = None
            self.original_feature_dim = 0
        else:
            self.features = pd.read_csv(features_path).values
            # 检查并处理NaN和Inf值
            nan_count = np.isnan(self.features).sum()
            inf_count = np.isinf(self.features).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"警告: 原始特征包含 {nan_count} 个NaN值和 {inf_count} 个Inf值，将替换为0")
                self.features = np.nan_to_num(self.features, nan=0.0, posinf=1.0, neginf=0.0)

            # 标准化特征
            self.features = (self.features - np.mean(self.features, axis=0)) / (np.std(self.features, axis=0) + 1e-8)

            self.original_feature_dim = self.features.shape[1]
            print(f"原始特征维度: {self.original_feature_dim}")

        # 数据增强选项
        self.augment = augment
        self.noise_level = 0.02

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        one_hot_sample = self.one_hot[idx]
        label = self.labels[idx]

        # 数据增强的几种处理
        if self.augment:
            if np.random.random() > 0.5:
                noise = np.random.normal(0, self.noise_level, one_hot_sample.shape)
                one_hot_sample = np.clip(one_hot_sample + noise, 0, 1)
            if np.random.random() > 0.5:
                # 序列重排
                one_hot_sample = np.random.permutation(one_hot_sample.T).T
            if np.random.random() > 0.5:
                # 碱基替换
                positions = np.random.choice(one_hot_sample.shape[1], size=int(one_hot_sample.shape[1] * 0.05), replace=False)
                for pos in positions:
                    one_hot_sample[:, pos] = np.eye(4)[np.random.randint(0, 4)]

        # 获取原始特征(如果存在)
        if self.features is not None:
            original_feature = torch.tensor(self.features[idx], dtype=torch.float32)
        else:
            original_feature = None

        one_hot_sample = torch.tensor(one_hot_sample, dtype=torch.float32).permute(1, 0)
        label = torch.tensor(label, dtype=torch.float32)  # 使用float32类型，因为是二分类标签

        if original_feature is not None:
            return one_hot_sample, original_feature, label
        else:
            return one_hot_sample, label