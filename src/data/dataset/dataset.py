import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# 定义路径
PROCESSED_PATH = r"D:\桌面\作业\genomic_data\data\data\preprocessed"
OUTPUT_PATH = r"D:\桌面\作业\genomic_data\data\data"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 创建子目录
for subset in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, subset), exist_ok=True)

# 加载数据
print("加载预处理数据...")
positive_seqs = np.loadtxt(os.path.join(PROCESSED_PATH, "positive_seqs.txt"), dtype=str)
negative_seqs = np.loadtxt(os.path.join(PROCESSED_PATH, "negative_seqs.txt"), dtype=str)

positive_one_hot = np.load(os.path.join(PROCESSED_PATH, "positive_one_hot.npy"), allow_pickle=True)
negative_one_hot = np.load(os.path.join(PROCESSED_PATH, "negative_one_hot.npy"), allow_pickle=True)

positive_features = pd.read_csv(os.path.join(PROCESSED_PATH, "positive_features.csv"))
negative_features = pd.read_csv(os.path.join(PROCESSED_PATH, "negative_features.csv"))

# 确保样本数量一致
assert len(positive_seqs) == len(positive_one_hot) == len(positive_features), "正样本数量不一致"
assert len(negative_seqs) == len(negative_one_hot) == len(negative_features), "负样本数量不一致"

# 创建标签 (1: 正样本, 0: 负样本)
positive_labels = np.ones(len(positive_seqs), dtype=int)
negative_labels = np.zeros(len(negative_seqs), dtype=int)

# 合并正负样本
all_seqs = np.concatenate([positive_seqs, negative_seqs])
all_one_hot = np.concatenate([positive_one_hot, negative_one_hot])
all_features = pd.concat([positive_features, negative_features], ignore_index=True)
all_labels = np.concatenate([positive_labels, negative_labels])

# 分层分割数据集
# 首先分离出测试集 (20%)
X_train_val, X_test, y_train_val, y_test, oh_train_val, oh_test, f_train_val, f_test = train_test_split(
    all_seqs, all_labels, all_one_hot, all_features, 
    test_size=0.2, random_state=42, stratify=all_labels
)

# 然后将剩余数据分为训练集和验证集 (70%/10%)
X_train, X_val, y_train, y_val, oh_train, oh_val, f_train, f_val = train_test_split(
    X_train_val, y_train_val, oh_train_val, f_train_val, 
    test_size=0.125, random_state=42, stratify=y_train_val
)  # 0.125 * 0.8 = 0.1 (10% of total)

# 保存数据集
def save_subset(subset_name, seqs, labels, one_hot, features):
    subset_dir = os.path.join(OUTPUT_PATH, subset_name)
    
    # 保存序列和标签
    np.savetxt(os.path.join(subset_dir, "sequences.txt"), seqs, fmt="%s")
    np.savetxt(os.path.join(subset_dir, "labels.txt"), labels, fmt="%d")
    
    # 保存One-hot编码
    np.save(os.path.join(subset_dir, "one_hot.npy"), one_hot)
    
    # 保存特征
    features.to_csv(os.path.join(subset_dir, "features.csv"), index=False)
    
    print(f"已保存 {subset_name} 集: {len(seqs)} 个样本")

# 保存各子集
save_subset("train", X_train, y_train, oh_train, f_train)
save_subset("validation", X_val, y_val, oh_val, f_val)
save_subset("test", X_test, y_test, oh_test, f_test)

# 复制原始特征文件用于参考
shutil.copy(os.path.join(PROCESSED_PATH, "positive_features.csv"), 
            os.path.join(OUTPUT_PATH, "positive_features_original.csv"))
shutil.copy(os.path.join(PROCESSED_PATH, "negative_features.csv"), 
            os.path.join(OUTPUT_PATH, "negative_features_original.csv"))

print(f"\n数据集分割完成！")
print(f"训练集: {len(X_train)} 个样本")
print(f"验证集: {len(X_val)} 个样本")
print(f"测试集: {len(X_test)} 个样本")
