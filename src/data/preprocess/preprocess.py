import os
import gzip
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
from itertools import product

# 路径配置（根据实际修改）
RAW_PATH = r"D:\桌面\作业\genomic_data\data\raw"  # 原始数据路径
PROCESSED_PATH = r"D:\桌面\作业\genomic_data\data\data\preprocessed"  # 预处理输出路径
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ========== 工具函数（覆盖流程图4~19步） ==========
def get_gc_content(seq: str) -> float:
    """步骤16：计算GC含量"""
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) if seq else 0.0

def one_hot_encode(seq: str) -> np.ndarray:
    """步骤12：一热编码（A/T/G/C→二进制向量）"""
    encoding = np.zeros((len(seq), 4))
    base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    for i, base in enumerate(seq):
        if base in base_map:
            encoding[i, base_map[base]] = 1
    return encoding

def calculate_kmer_features(seq: str, k_list: list) -> dict:
    """步骤13：k-mer频率特征（计算不同长度k-mer频率）"""
    kmer_features = {}
    valid_bases = {'A', 'T', 'G', 'C'}
    for k in k_list:
        # 初始化所有可能的k-mer（避免特征缺失）
        for bases in product(valid_bases, repeat=k):
            kmer = ''.join(bases)
            kmer_features[f"kmer_{k}_{kmer}"] = 0
        
        # 统计实际k-mer频率
        max_pos = len(seq) - k + 1
        for i in range(max_pos):
            kmer = seq[i:i+k]
            if all(base in valid_bases for base in kmer):
                kmer_features[f"kmer_{k}_{kmer}"] += 1
        
        # 归一化（避免长度偏差）
        total_kmers = max(1, max_pos)  # 防止除零
        for key in list(kmer_features.keys()):
            if key.startswith(f"kmer_{k}_"):
                kmer_features[key] /= total_kmers
    return kmer_features

def calculate_position_features(seq: str, max_len: int = 200) -> dict:
    """步骤14：位置特异性特征（记录每个位置的碱基）"""
    position_features = {}
    valid_bases = {'A', 'T', 'G', 'C'}
    for i in range(max_len):
        # 初始化所有碱基为0
        for base in valid_bases:
            position_features[f"pos_{i}_{base}"] = 0
        # 超出序列长度时用'N'占位（避免KeyError）
        if i >= len(seq):
            position_features[f"pos_{i}_N"] = 1  # 标记超出位置
            continue
        # 记录实际碱基
        base = seq[i]
        if base in valid_bases:
            position_features[f"pos_{i}_{base}"] = 1
    return position_features

def sequence_complexity(seq: str) -> float:
    """步骤19：序列复杂度（香农熵）"""
    from math import log2
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
    """步骤15-19：整合所有特征工程"""
    features = {
        # 步骤16
        "gc_content": get_gc_content(seq),
        # 步骤17
        "cpg_dinuc_freq": seq.count('CG') / (len(seq) - 1) if len(seq) > 1 else 0.0,
        # 步骤18
        "obs_exp_ratio": 0.0,
        # 步骤19
        "seq_complexity": sequence_complexity(seq)
    }
    
    # 计算Obs/Exp比率（步骤18）
    c_count = seq.count('C')
    g_count = seq.count('G')
    if c_count > 0 and g_count > 0 and len(seq) > 1:
        obs = seq.count('CG')
        exp = (c_count * g_count) / (len(seq) - 1)
        features["obs_exp_ratio"] = obs / exp if exp != 0 else 0.0
    
    # 步骤13：k-mer特征（k=3,4）
    kmer_features = calculate_kmer_features(seq, k_list=[3, 4])
    features.update(kmer_features)
    
    # 步骤14：位置特征（固定窗口200bp）
    position_features = calculate_position_features(seq, max_len=200)
    features.update(position_features)
    
    return features

def sliding_window_split(sequences: list, window: int = 200, step: int = 100) -> list:
    """步骤8-10：滑动窗口分割（固定长度+滑动增加多样性）"""
    split_seqs = []
    for seq in sequences:
        # 仅处理有效长度序列
        if len(seq) < window:
            continue
        # 滑动窗口分割
        for i in range(0, len(seq) - window + 1, step):
            split_seqs.append(seq[i:i+window])
    return split_seqs

# ========== 样本生成核心逻辑（覆盖流程图4-7步） ==========
def get_cpg_regions(cpg_path: str, chrom: str) -> list:
    """步骤3-6：从BED文件提取指定染色体的CpG岛区域"""
    cpg_regions = []
    open_func = gzip.open if cpg_path.endswith('.gz') else open
    with open_func(cpg_path, "rt") as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) >= 3 and cols[0] == chrom:
                cpg_regions.append((int(cols[1]), int(cols[2])))
    return cpg_regions

def extract_positive_samples(chrom_seq: str, cpg_regions: list, window: int) -> list:
    """步骤5：从CpG岛区域提取正样本"""
    positive = []
    for start, end in cpg_regions:
        # 确保区域在染色体范围内
        if end > len(chrom_seq):
            continue
        region_seq = chrom_seq[start:end]
        # 滑动窗口分割（同步骤8-10）
        for i in range(0, len(region_seq) - window + 1, window//2):  # 步长设为窗口一半增加多样性
            positive.append(region_seq[i:i+window])
    return positive

def extract_negative_samples(chrom_seq: str, cpg_regions: list, window: int, positive_count: int) -> list:
    """步骤6：从非CpG岛区域提取负样本（平衡数量）"""
    # 标记CpG岛区域
    is_cpg = np.zeros(len(chrom_seq), dtype=bool)
    for start, end in cpg_regions:
        if end <= len(chrom_seq):
            is_cpg[start:end] = True
    
    # 仅在非CpG区域采样
    negative = []
    attempts = 0
    max_attempts = positive_count * 10  # 最多尝试10倍次数
    while len(negative) < positive_count and attempts < max_attempts:
        attempts += 1
        start = random.randint(0, len(chrom_seq) - window)
        end_pos = start + window
        # 检查是否完全在非CpG区域
        if not np.any(is_cpg[start:end_pos]):
            negative.append(chrom_seq[start:end_pos])
    return negative[:positive_count]  # 截断到正样本数量

def process_chromosome(chrom: str, hg38_path: str, cpg_path: str, window: int = 200) -> tuple:
    """步骤4-7：处理单条染色体，生成正负样本"""
    print(f"开始处理染色体 {chrom}...")
    
    # 步骤2-3：加载参考基因组和CpG注释
    chrom_seq = None
    for record in SeqIO.parse(hg38_path, "fasta"):
        if record.id == chrom:
            chrom_seq = str(record.seq).upper()
            break
    if chrom_seq is None:
        print(f"染色体 {chrom} 不存在于参考基因组，跳过")
        return [], []
    
    cpg_regions = get_cpg_regions(cpg_path, chrom)
    print(f"  找到 {len(cpg_regions)} 个CpG岛区域")
    
    # 步骤5：生成正样本
    positive = extract_positive_samples(chrom_seq, cpg_regions, window)
    print(f"  正样本原始数量: {len(positive)}")
    
    # 步骤6-7：生成负样本并平衡数量
    negative = extract_negative_samples(chrom_seq, cpg_regions, window, len(positive))
    print(f"  负样本平衡后数量: {len(negative)}")
    
    return positive, negative

# ========== 主流程（整合所有步骤） ==========
def main():
    # 步骤2-3：检查输入文件
    hg38_files = [f for f in os.listdir(RAW_PATH) if f.startswith("hg38.fa")]
    cpg_files = [f for f in os.listdir(RAW_PATH) if f.startswith("cpgIslandExt") and (f.endswith(".bed") or f.endswith(".bed.gz"))]
    if not hg38_files or not cpg_files:
        raise FileNotFoundError("缺少参考基因组或CpG注释文件！")
    hg38_path = os.path.join(RAW_PATH, hg38_files[0])
    cpg_path = os.path.join(RAW_PATH, cpg_files[0])
    
    # 步骤4：处理所有染色体（可自定义染色体列表，如仅处理常染色体）
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]  # 人类主要染色体
    all_positive, all_negative = [], []
    
    for chrom in chromosomes:
        positive, negative = process_chromosome(chrom, hg38_path, cpg_path, window=200)
        all_positive.extend(positive)
        all_negative.extend(negative)
    
    # 步骤7：最终样本平衡
    min_size = min(len(all_positive), len(all_negative))
    all_positive = all_positive[:min_size]
    all_negative = all_negative[:min_size]
    print(f"\n全局样本平衡后：正样本 {len(all_positive)} | 负样本 {len(all_negative)}")
    
    # 步骤8-10：滑动窗口分割
    print("开始序列分割（滑动窗口）...")
    positive_split = sliding_window_split(all_positive, window=200, step=100)
    negative_split = sliding_window_split(all_negative, window=200, step=100)
    print(f"分割后：正样本 {len(positive_split)} | 负样本 {len(negative_split)}")
    
    # 步骤11-14：序列编码（一热编码）
    print("开始序列编码（一热编码）...")
    positive_one_hot = [one_hot_encode(seq) for seq in positive_split]
    negative_one_hot = [one_hot_encode(seq) for seq in negative_split]
    
    # 步骤15-19：特征工程
    print("开始特征工程（GC含量、k-mer、位置特征等）...")
    positive_features = [calculate_all_features(seq) for seq in positive_split]
    negative_features = [calculate_all_features(seq) for seq in negative_split]
    
    # 保存结果（确保覆盖流程图所有输出）
    print("开始保存结果...")
    # 序列文本
    with open(os.path.join(PROCESSED_PATH, "positive_seqs.txt"), "w") as f:
        f.write("\n".join(positive_split))
    with open(os.path.join(PROCESSED_PATH, "negative_seqs.txt"), "w") as f:
        f.write("\n".join(negative_split))
    # 一热编码（Numpy二进制）
    np.save(os.path.join(PROCESSED_PATH, "positive_one_hot.npy"), np.array(positive_one_hot, dtype=object))
    np.save(os.path.join(PROCESSED_PATH, "negative_one_hot.npy"), np.array(negative_one_hot, dtype=object))
    # 特征工程（CSV表格）
    pd.DataFrame(positive_features).to_csv(os.path.join(PROCESSED_PATH, "positive_features.csv"), index=False)
    pd.DataFrame(negative_features).to_csv(os.path.join(PROCESSED_PATH, "negative_features.csv"), index=False)
    
    print("预处理完成！所有步骤严格覆盖流程图4~19步")

if __name__ == "__main__":
    main()