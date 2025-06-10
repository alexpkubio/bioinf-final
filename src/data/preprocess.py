import os
import random
import gzip
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict

# 路径配置
RAW_PATH = r"D:\桌面\作业\genomic_data\data\raw"
PROCESSED_PATH = r"D:\桌面\作业\genomic_data\data\processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ========== 1-3. 序列获取（从本地文件加载，不再包含下载逻辑） ==========
def load_genome_fasta(fasta_path):
    """加载基因组序列（支持 .fa 或 .fa.gz 格式）"""
    if fasta_path.endswith('.gz'):
        open_func = lambda x: gzip.open(x, 'rt')
    else:
        open_func = lambda x: open(x, 'r')
        
    with open_func(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            yield record.id, str(record.seq).upper()

def load_cpg_bed(bed_path):
    """加载 CpG 岛注释（支持 .bed 或 .txt.gz 格式）"""
    if bed_path.endswith('.gz'):
        open_func = lambda x: gzip.open(x, 'rt')
    else:
        open_func = lambda x: open(x, 'r')
        
    cpg_regions = []
    with open_func(bed_path) as f:
        for line in f:
            cols = line.strip().split()
            # 假设 BED 格式：chrom, start, end, ...
            if len(cols) >= 3:
                cpg_regions.append((cols[0], int(cols[1]), int(cols[2])))
    return cpg_regions

# ========== 4-7. 正负样本生成（核心：负样本匹配 GC 含量） ==========
def get_gc_content(seq):
    """计算序列 GC 含量"""
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq) if len(seq) > 0 else 0

def extract_positive_samples(genome, cpg_regions, window=200):
    """提取正样本（CpG 岛区域）"""
    positive = []
    for chrom, start, end in cpg_regions:
        if chrom not in genome:
            continue
        seq = genome[chrom][start:end]
        # 按 window 分割（若区域长度不足，可调整或跳过）
        for i in range(0, len(seq), window):
            sub_seq = seq[i:i+window]
            if len(sub_seq) == window:
                positive.append(sub_seq)
    return positive

def extract_negative_samples(genome, cpg_regions, positive_gc_stats, window=200, gc_tolerance=0.1, max_tries=10):
    """提取负样本（非 CpG 岛 + GC 含量匹配）"""
    negative = []
    # 先标记 CpG 岛区域，避免重复选取
    cpg_mask = defaultdict(lambda: defaultdict(bool))
    for chrom, start, end in cpg_regions:
        for pos in range(start, end):
            cpg_mask[chrom][pos] = True

    # 获取正样本 GC 统计信息
    mean_gc, std_gc = positive_gc_stats
    
    # 遍历基因组，找非 CpG 岛区域
    for chrom, seq in genome.items():
        chrom_len = len(seq)
        # 滑动窗口找候选区域
        for start in range(0, chrom_len - window, window):
            end = start + window
            # 检查是否与 CpG 岛区域重叠
            overlap = any(cpg_mask[chrom].get(pos, False) for pos in range(start, end))
            if overlap:
                continue
            # 计算候选区域 GC 含量
            gc = get_gc_content(seq[start:end])
            # 匹配正样本 GC 分布（改进版：使用均值和标准差）
            if (mean_gc - std_gc * gc_tolerance) <= gc <= (mean_gc + std_gc * gc_tolerance):
                negative.append(seq[start:end])
                
    return negative

# ========== 8-10. 序列分割（滑动窗口） ==========
def sliding_window(seq, window=200, step=100):
    """滑动窗口分割序列（内存优化：生成器）"""
    for i in range(0, len(seq) - window + 1, step):
        yield seq[i:i+window]

# ========== 11-14. 序列编码（One-hot + k-mer + 位置特征） ==========
BASES = ["A", "T", "G", "C"]
BASE_TO_IDX = {base: i for i, base in enumerate(BASES)}

def one_hot_encode(seq):
    """One-hot 编码"""
    encoding = np.zeros((len(seq), len(BASES)))
    for i, base in enumerate(seq):
        if base in BASE_TO_IDX:
            encoding[i, BASE_TO_IDX[base]] = 1
    return encoding

def kmer_frequency(seq, k=3):
    """计算 k-mer 频率"""
    kmer_counts = defaultdict(int)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmer_counts[kmer] += 1
    # 转换为频率
    total = len(seq) - k + 1
    return {kmer: cnt / total for kmer, cnt in kmer_counts.items()}

# ========== 15-19. 特征工程（GC、CpG 二核苷酸等） ==========
def calculate_features(seq):
    """计算所有特征工程指标"""
    gc = get_gc_content(seq)
    cpg_dinuc = seq.count("CG") / (len(seq) - 1) if len(seq) > 1 else 0
    # Obs/Exp 计算
    c_count = seq.count("C")
    g_count = seq.count("G")
    exp = (c_count * g_count) / (len(seq) - 1) if (len(seq) > 1 and c_count * g_count != 0) else 0
    obs_exp = cpg_dinuc / exp if exp != 0 else 0
    # 序列复杂度（简化为 Shannon 熵）
    from math import log2
    counts = defaultdict(int)
    for base in seq:
        counts[base] += 1
    entropy = -sum((cnt / len(seq)) * log2(cnt / len(seq)) for cnt in counts.values()) if len(seq) > 0 else 0
    return {
        "gc_content": gc,
        "cpg_dinucleotide_freq": cpg_dinuc,
        "obs_exp_ratio": obs_exp,
        "sequence_complexity": entropy
    }

# ========== 主流程 ==========
def main():
    # 检查原始文件是否存在
    hg38_files = [f for f in os.listdir(RAW_PATH) if f.startswith('hg38.fa')]
    cpg_files = [f for f in os.listdir(RAW_PATH) if f.startswith('cpgIslandExt') and (f.endswith('.bed') or f.endswith('.txt.gz'))]
    
    if not hg38_files:
        raise FileNotFoundError("未找到 hg38.fa 或 hg38.fa.gz 文件")
    if not cpg_files:
        raise FileNotFoundError("未找到 cpgIslandExt.bed 或 cpgIslandExt.txt.gz 文件")
    
    # 使用找到的文件路径
    hg38_path = os.path.join(RAW_PATH, hg38_files[0])
    cpg_path = os.path.join(RAW_PATH, cpg_files[0])
    
    print(f"使用基因组文件: {hg38_path}")
    print(f"使用 CpG 注释文件: {cpg_path}")
    
    # 1. 加载数据
    print("正在加载基因组序列...")
    genome = {record_id: seq for record_id, seq in load_genome_fasta(hg38_path)}
    print("正在加载 CpG 岛注释...")
    cpg_regions = load_cpg_bed(cpg_path)
    
    # 2. 生成正样本
    print("正在生成正样本...")
    positive = extract_positive_samples(genome, cpg_regions, window=200)
    print(f"正样本数量: {len(positive)}")
    
    # 3. 计算正样本 GC 统计信息
    print("正在计算正样本 GC 含量分布...")
    positive_gc_values = [get_gc_content(seq) for seq in positive]
    mean_gc = np.mean(positive_gc_values)
    std_gc = np.std(positive_gc_values)
    print(f"正样本 GC 含量均值: {mean_gc:.4f}, 标准差: {std_gc:.4f}")
    
    # 4. 生成负样本（改进：使用 GC 统计信息而非直接引用 positive 变量）
    print("正在生成负样本（匹配 GC 含量）...")
    negative = extract_negative_samples(genome, cpg_regions, (mean_gc, std_gc), window=200, gc_tolerance=0.1)
    print(f"负样本数量: {len(negative)}")
    
    # 5. 平衡样本数量
    min_size = min(len(positive), len(negative))
    positive = positive[:min_size]
    negative = negative[:min_size]
    print(f"平衡后样本数量: {min_size}")
    
    # 6. 保存正负样本
    with open(os.path.join(PROCESSED_PATH, "positive_samples.txt"), "w") as f:
        f.write("\n".join(positive))
    with open(os.path.join(PROCESSED_PATH, "negative_samples.txt"), "w") as f:
        f.write("\n".join(negative))
    
    # 7. 滑动窗口扩展样本
    print("正在进行滑动窗口分割...")
    positive_windows = []
    for seq in positive:
        positive_windows.extend(sliding_window(seq, window=200, step=100))
    
    negative_windows = []
    for seq in negative:
        negative_windows.extend(sliding_window(seq, window=200, step=100))
    
    # 8. 保存分割后样本
    with open(os.path.join(PROCESSED_PATH, "positive_windows.txt"), "w") as f:
        f.write("\n".join(positive_windows))
    with open(os.path.join(PROCESSED_PATH, "negative_windows.txt"), "w") as f:
        f.write("\n".join(negative_windows))
    
    # 9. 序列编码
    print("正在进行序列编码...")
    positive_encoded = [one_hot_encode(seq) for seq in positive_windows]
    negative_encoded = [one_hot_encode(seq) for seq in negative_windows]
    
    # 保存编码
    np.save(os.path.join(PROCESSED_PATH, "positive_encoded.npy"), np.array(positive_encoded, dtype=object))
    np.save(os.path.join(PROCESSED_PATH, "negative_encoded.npy"), np.array(negative_encoded, dtype=object))
    
    # 10. 特征工程
    print("正在计算特征工程指标...")
    positive_features = [calculate_features(seq) for seq in positive_windows]
    negative_features = [calculate_features(seq) for seq in negative_windows]
    
    # 保存为 DataFrame
    pd.DataFrame(positive_features).to_csv(os.path.join(PROCESSED_PATH, "positive_features.csv"), index=False)
    pd.DataFrame(negative_features).to_csv(os.path.join(PROCESSED_PATH, "negative_features.csv"), index=False)
    
    print("=== 数据预处理全流程完成 ===")
    print(f"最终正样本数量: {len(positive_windows)}, 负样本数量: {len(negative_windows)}")
    print(f"输出路径: {PROCESSED_PATH}")

if __name__ == "__main__":
    main()