#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
from itertools import product
import re
import h5py
import gc  # 显式垃圾回收

# 路径配置（根据实际修改）
INPUT_DIR = r"D:\桌面\作业\genomic_data\data\data\expanded_dataset_20250616_111848"  # 输入数据路径
PROCESSED_PATH = os.path.join(INPUT_DIR, "preprocessed_features")  # 预处理输出路径
os.makedirs(PROCESSED_PATH, exist_ok=True)

# 内存优化配置
BATCH_SIZE = 1000  # 每批处理的序列数

# ========== 工具函数 ==========
def get_gc_content(seq: str) -> float:
    """计算GC含量"""
    if not seq:
        return 0.0
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

def one_hot_encode(seq: str) -> np.ndarray:
    """一热编码（A/T/G/C→二进制向量）"""
    encoding = np.zeros((200, 4), dtype=np.float32)  # 固定长度为200
    base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    for i, base in enumerate(seq[:200]):  # 截断超过200bp的序列
        if base in base_map:
            encoding[i, base_map[base]] = 1
    return encoding

def calculate_kmer_features(seq: str, k_list: list = [2, 3, 4]) -> dict:
    """k-mer频率特征（包含k=2）"""
    kmer_features = {}
    valid_bases = {'A', 'T', 'G', 'C'}
    
    for k in k_list:
        # 初始化计数器
        kmer_counts = defaultdict(int)
        
        # 统计k-mer
        max_pos = len(seq) - k + 1
        for i in range(max_pos):
            kmer = seq[i:i+k]
            if all(base in valid_bases for base in kmer):
                kmer_counts[kmer] += 1
        
        # 归一化并存储
        total_kmers = max(1, max_pos)
        for bases in product(valid_bases, repeat=k):
            kmer = ''.join(bases)
            kmer_features[f"kmer_{k}_{kmer}"] = kmer_counts[kmer] / total_kmers
    
    return kmer_features

def calculate_position_features(seq: str, max_len: int = 200) -> dict:
    """位置特异性特征（优化版：稀疏表示）"""
    position_features = {}
    valid_bases = {'A', 'T', 'G', 'C'}
    
    # 仅记录实际存在的碱基
    for i, base in enumerate(seq[:max_len]):
        if base in valid_bases:
            position_features[f"pos_{i}_{base}"] = 1
    
    # 添加序列长度特征
    position_features["seq_length"] = len(seq)
    
    return position_features

def sequence_complexity(seq: str) -> float:
    """序列复杂度（香农熵）"""
    from math import log2
    if not seq:
        return 0.0
    
    base_count = defaultdict(int)
    for base in seq:
        base_count[base] += 1
    
    entropy = 0.0
    total = len(seq)
    for cnt in base_count.values():
        if cnt == 0:
            continue
        prob = cnt / total
        entropy -= prob * log2(prob)
    
    return entropy

def calculate_all_features(seq: str) -> dict:
    """整合所有特征工程（包含k=2的k-mer）"""
    features = {
        "gc_content": get_gc_content(seq),
        "cpg_dinuc_freq": seq.count('CG') / (len(seq) - 1) if len(seq) > 1 else 0.0,
        "obs_exp_ratio": 0.0,
        "seq_complexity": sequence_complexity(seq)
    }
    
    # 计算Obs/Exp比率
    c_count = seq.count('C')
    g_count = seq.count('G')
    if c_count > 0 and g_count > 0 and len(seq) > 1:
        obs = seq.count('CG')
        exp = (c_count * g_count) / (len(seq) - 1)
        features["obs_exp_ratio"] = obs / exp if exp != 0 else 0.0
    
    # k-mer特征（包含k=2）
    features.update(calculate_kmer_features(seq, k_list=[2, 3, 4]))
    
    # 位置特征（稀疏表示）
    features.update(calculate_position_features(seq))
    
    return features

def sliding_window_split(sequences: list, window: int = 200, step: int = 100) -> list:
    """滑动窗口分割（优化版）"""
    for seq in sequences:
        if len(seq) < window:
            yield seq  # 短序列直接返回
        else:
            for i in range(0, len(seq) - window + 1, step):
                yield seq[i:i+window]

def calculate_and_save_features(sequences, output_file, process_fn, chunk_size=BATCH_SIZE):
    """流式处理序列特征并直接写入文件"""
    header_written = False
    total_seqs = len(sequences)
    
    for i in range(0, total_seqs, chunk_size):
        batch = sequences[i:min(i+chunk_size, total_seqs)]
        
        # 使用生成器处理，减少内存峰值
        batch_features = (process_fn(seq) for seq in batch)
        
        # 转换为DataFrame并追加到CSV
        df = pd.DataFrame(batch_features)
        df.to_csv(output_file, mode='a', index=False, header=(not header_written))
        header_written = True
        
        # 强制垃圾回收
        del df, batch, batch_features
        gc.collect()
        
        print(f"已处理 {min(i+chunk_size, total_seqs)}/{total_seqs} 条序列")

# ========== 主流程 ==========
def main():
    # 检查输入文件是否存在
    positive_file = os.path.join(INPUT_DIR, "positive_samples.txt")
    negative_file = os.path.join(INPUT_DIR, "negative_samples.txt")
    
    if not os.path.exists(positive_file) or not os.path.exists(negative_file):
        raise FileNotFoundError(f"输入文件不存在！请检查 {positive_file} 和 {negative_file}")
    
    # 读取已生成的正负样本（使用生成器减少内存占用）
    print("读取已生成的正负样本序列...")
    with open(positive_file, "r") as f:
        all_positive = [line.strip() for line in f if line.strip()]
    
    with open(negative_file, "r") as f:
        all_negative = [line.strip() for line in f if line.strip()]
    
    print(f"读取完成：正样本 {len(all_positive)} | 负样本 {len(all_negative)}")
    
    # 步骤：滑动窗口分割（使用生成器）
    print("开始序列分割（滑动窗口）...")
    positive_split = list(sliding_window_split(all_positive))
    negative_split = list(sliding_window_split(all_negative))
    print(f"分割后：正样本 {len(positive_split)} | 负样本 {len(negative_split)}")
    
    # 检查分割后是否有数据
    if len(positive_split) == 0:
        print("警告：分割后的正样本数据为空，跳过正样本编码")
    if len(negative_split) == 0:
        print("警告：分割后的负样本数据为空，跳过负样本编码")
    
    # 步骤：序列编码（一热编码） - 直接写入HDF5
    print("开始序列编码（一热编码）...")
    with h5py.File(os.path.join(PROCESSED_PATH, "one_hot_encodings.h5"), 'w') as hf:
        # 创建可扩展数据集
        dset_pos = hf.create_dataset("positive", shape=(0, 200, 4), maxshape=(None, 200, 4), dtype=np.float32, chunks=True)
        dset_neg = hf.create_dataset("negative", shape=(0, 200, 4), maxshape=(None, 200, 4), dtype=np.float32, chunks=True)
        
        # 分批处理正样本
        if len(positive_split) > 0:
            for i in range(0, len(positive_split), BATCH_SIZE):
                batch = positive_split[i:i+BATCH_SIZE]
                batch_encoded = np.array([one_hot_encode(seq) for seq in batch])
                dset_pos.resize(dset_pos.shape[0] + len(batch), axis=0)
                dset_pos[-len(batch):] = batch_encoded
                
                # 打印进度前先保存batch长度，避免删除后引用
                batch_len = len(batch)
                del batch, batch_encoded
                gc.collect()
                
                # 使用保存的batch长度
                print(f"已编码正样本 {i+batch_len}/{len(positive_split)}")
        
        # 分批处理负样本
        if len(negative_split) > 0:
            for i in range(0, len(negative_split), BATCH_SIZE):
                batch = negative_split[i:i+BATCH_SIZE]
                batch_encoded = np.array([one_hot_encode(seq) for seq in batch])
                dset_neg.resize(dset_neg.shape[0] + len(batch), axis=0)
                dset_neg[-len(batch):] = batch_encoded
                
                batch_len = len(batch)
                del batch, batch_encoded
                gc.collect()
                
                print(f"已编码负样本 {i+batch_len}/{len(negative_split)}")
    
    # 步骤：特征工程 - 直接写入CSV
    print("开始特征工程（GC含量、k-mer、位置特征等）...")
    if len(positive_split) > 0:
        print("处理正样本特征...")
        calculate_and_save_features(positive_split, os.path.join(PROCESSED_PATH, "positive_features.csv"), calculate_all_features)
    
    if len(negative_split) > 0:
        print("处理负样本特征...")
        calculate_and_save_features(negative_split, os.path.join(PROCESSED_PATH, "negative_features.csv"), calculate_all_features)
    
    # 保存序列文本（可选）
    print("保存序列文本...")
    if len(positive_split) > 0:
        with open(os.path.join(PROCESSED_PATH, "positive_seqs.txt"), "w") as f:
            for seq in positive_split:
                f.write(seq + "\n")
    
    if len(negative_split) > 0:
        with open(os.path.join(PROCESSED_PATH, "negative_seqs.txt"), "w") as f:
            for seq in negative_split:
                f.write(seq + "\n")
    
    print(f"预处理完成！结果已保存至 {PROCESSED_PATH}")

if __name__ == "__main__":
    main()