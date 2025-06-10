import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 路径配置
PROCESSED_PATH = r"D:\桌面\作业\genomic_data\data\processed"
RESULTS_PATH = r"D:\桌面\作业\genomic_data\data\results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# 定义分割比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def load_data():
    """加载预处理好的正负样本数据"""
    print("正在加载预处理数据...")
    
    # 加载序列数据
    with open(os.path.join(PROCESSED_PATH, "positive_windows.txt")) as f:
        positive_seqs = [line.strip() for line in f]
    
    with open(os.path.join(PROCESSED_PATH, "negative_windows.txt")) as f:
        negative_seqs = [line.strip() for line in f]
    
    # 加载特征数据
    positive_features = pd.read_csv(os.path.join(PROCESSED_PATH, "positive_features.csv"))
    negative_features = pd.read_csv(os.path.join(PROCESSED_PATH, "negative_features.csv"))
    
    # 加载编码数据
    positive_encoded = np.load(os.path.join(PROCESSED_PATH, "positive_encoded.npy"), allow_pickle=True)
    negative_encoded = np.load(os.path.join(PROCESSED_PATH, "negative_encoded.npy"), allow_pickle=True)
    
    print(f"正样本数量: {len(positive_seqs)}, 负样本数量: {len(negative_seqs)}")
    return positive_seqs, negative_seqs, positive_features, negative_features, positive_encoded, negative_encoded

def split_indices(n_samples, train_ratio, val_ratio, test_ratio, random_state=42):
    """分割样本索引为训练、验证和测试集"""
    # 确保比例之和为1
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1.0"
    
    # 先分割训练集和剩余集
    train_idx, temp_idx = train_test_split(
        range(n_samples), 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    # 计算验证集的比例（相对于剩余集）
    val_ratio_remaining = val_ratio / (val_ratio + test_ratio)
    
    # 分割剩余集为验证集和测试集
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_ratio_remaining, 
        random_state=random_state
    )
    
    return train_idx, val_idx, test_idx

def split_and_save_data(positive_data, negative_data, data_type, random_state=42):
    """分割并保存各类数据"""
    # 确保正负样本数量相同
    assert len(positive_data) == len(negative_data), "正负样本数量不一致"
    n_samples = len(positive_data)
    
    # 获取分割索引
    train_idx, val_idx, test_idx = split_indices(
        n_samples, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO, 
        random_state=random_state
    )
    
    # 创建标签（1为正样本，0为负样本）
    positive_labels = np.ones(len(positive_data))
    negative_labels = np.zeros(len(negative_data))
    
    # 合并正负样本
    all_data = np.concatenate([positive_data, negative_data])
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # 将索引转换为numpy数组
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    # 生成负样本的索引（正样本索引 + n_samples）
    neg_train_idx = train_idx + n_samples
    neg_val_idx = val_idx + n_samples
    neg_test_idx = test_idx + n_samples
    
    # 分割数据
    train_data = all_data[np.concatenate([train_idx, neg_train_idx])]
    val_data = all_data[np.concatenate([val_idx, neg_val_idx])]
    test_data = all_data[np.concatenate([test_idx, neg_test_idx])]
    
    # 分割标签
    train_labels = all_labels[np.concatenate([train_idx, neg_train_idx])]
    val_labels = all_labels[np.concatenate([val_idx, neg_val_idx])]
    test_labels = all_labels[np.concatenate([test_idx, neg_test_idx])]
    
    # 保存数据
    save_path = os.path.join(RESULTS_PATH, data_type)
    os.makedirs(save_path, exist_ok=True)
    
    if data_type == "sequences":
        # 保存序列数据为文本文件
        with open(os.path.join(save_path, "train.txt"), "w") as f:
            for seq, label in zip(train_data, train_labels):
                f.write(f"{seq}\t{int(label)}\n")
        
        with open(os.path.join(save_path, "val.txt"), "w") as f:
            for seq, label in zip(val_data, val_labels):
                f.write(f"{seq}\t{int(label)}\n")
        
        with open(os.path.join(save_path, "test.txt"), "w") as f:
            for seq, label in zip(test_data, test_labels):
                f.write(f"{seq}\t{int(label)}\n")
                
    elif data_type == "features":
        # 保存特征数据为CSV
        train_df = pd.DataFrame(train_data)
        train_df['label'] = train_labels
        train_df.to_csv(os.path.join(save_path, "train.csv"), index=False)
        
        val_df = pd.DataFrame(val_data)
        val_df['label'] = val_labels
        val_df.to_csv(os.path.join(save_path, "val.csv"), index=False)
        
        test_df = pd.DataFrame(test_data)
        test_df['label'] = test_labels
        test_df.to_csv(os.path.join(save_path, "test.csv"), index=False)
        
    elif data_type == "encoded":
        # 保存编码数据为numpy文件
        np.save(os.path.join(save_path, "train.npy"), train_data)
        np.save(os.path.join(save_path, "val.npy"), val_data)
        np.save(os.path.join(save_path, "test.npy"), test_data)
        
        # 保存标签
        np.save(os.path.join(save_path, "train_labels.npy"), train_labels)
        np.save(os.path.join(save_path, "val_labels.npy"), val_labels)
        np.save(os.path.join(save_path, "test_labels.npy"), test_labels)
    
    print(f"{data_type} 分割完成:")
    print(f"  训练集: {len(train_data)} 样本")
    print(f"  验证集: {len(val_data)} 样本")
    print(f"  测试集: {len(test_data)} 样本")
    
    return len(train_data), len(val_data), len(test_data)

def main():
    print(f"开始数据分割，结果将保存至: {RESULTS_PATH}")
    
    # 加载数据
    positive_seqs, negative_seqs, positive_features, negative_features, positive_encoded, negative_encoded = load_data()
    
    # 分割并保存序列数据
    seq_train_size, seq_val_size, seq_test_size = split_and_save_data(
        positive_seqs, 
        negative_seqs, 
        "sequences"
    )
    
    # 分割并保存特征数据
    feat_train_size, feat_val_size, feat_test_size = split_and_save_data(
        positive_features.to_numpy(), 
        negative_features.to_numpy(), 
        "features"
    )
    
    # 分割并保存编码数据
    enc_train_size, enc_val_size, enc_test_size = split_and_save_data(
        positive_encoded, 
        negative_encoded, 
        "encoded"
    )
    
    # 验证所有分割结果一致
    assert (seq_train_size == feat_train_size == enc_train_size and
            seq_val_size == feat_val_size == enc_val_size and
            seq_test_size == feat_test_size == enc_test_size), "分割结果不一致!"
    
    # 保存分割统计信息
    stats = {
        "total_samples": seq_train_size + seq_val_size + seq_test_size,
        "train_samples": seq_train_size,
        "val_samples": seq_val_size,
        "test_samples": seq_test_size,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(RESULTS_PATH, "split_statistics.csv"), index=False)
    
    print("\n=== 数据分割完成 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"训练集: {stats['train_samples']} ({stats['train_ratio']*100:.1f}%)")
    print(f"验证集: {stats['val_samples']} ({stats['val_ratio']*100:.1f}%)")
    print(f"测试集: {stats['test_samples']} ({stats['test_ratio']*100:.1f}%)")
    print(f"统计信息已保存至: {os.path.join(RESULTS_PATH, 'split_statistics.csv')}")

if __name__ == "__main__":
    main()