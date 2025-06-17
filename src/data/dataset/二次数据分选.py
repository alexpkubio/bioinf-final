import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import h5py
import gc

# 定义输入路径
PROCESSED_PATH = r"D:\桌面\作业\genomic_data\data\data\expanded_dataset_20250616_111848\preprocessed_features"

# 定义输出路径
OUTPUT_BASE_DIR = r"D:\桌面\作业\genomic_data\data\data\expanded_dataset_20250616_111848"
OUTPUT_PATH = os.path.join(OUTPUT_BASE_DIR, "split_dataset")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 创建子目录
for subset in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, subset), exist_ok=True)

# 辅助函数：追加保存序列和标签
def append_to_text_file(file_path, data, fmt):
    with open(file_path, 'a') as f:
        np.savetxt(f, data, fmt=fmt, newline='\n')

# 辅助函数：获取数据集行数
def get_csv_row_count(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f) - 1  # 减1是因为有标题行

# 处理一热编码数据（分批处理）
def process_one_hot_data():
    print("开始处理一热编码数据...")
    
    # 创建输出HDF5文件
    output_h5_path = os.path.join(OUTPUT_PATH, "one_hot_encodings.h5")
    with h5py.File(output_h5_path, 'w') as out_hf:
        # 创建可扩展数据集
        dset_config = {
            'chunks': True,
            'maxshape': (None, 200, 4),
            'dtype': np.float32
        }
        
        # 为训练集、验证集和测试集创建数据集
        train_dset = out_hf.create_dataset("train", shape=(0, 200, 4), **dset_config)
        val_dset = out_hf.create_dataset("validation", shape=(0, 200, 4), **dset_config)
        test_dset = out_hf.create_dataset("test", shape=(0, 200, 4), **dset_config)
        
        # 处理正样本
        with h5py.File(os.path.join(PROCESSED_PATH, "one_hot_encodings.h5"), 'r') as hf:
            positive_dset = hf['positive']
            total_samples = len(positive_dset)
            batch_size = 500  # 根据内存情况调整批次大小
            
            print(f"处理正样本: {total_samples} 条记录")
            
            for i in range(0, total_samples, batch_size):
                # 读取批次数据
                batch = positive_dset[i:i+batch_size].astype(np.float32)
                labels = np.ones(len(batch), dtype=int)
                
                # 分割批次数据
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    batch, labels, test_size=0.2, random_state=42, stratify=labels
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
                )
                
                # 追加到输出数据集
                def append_to_dset(dset, data):
                    current_size = dset.shape[0]
                    dset.resize(current_size + len(data), axis=0)
                    dset[current_size:] = data
                
                append_to_dset(train_dset, X_train)
                append_to_dset(val_dset, X_val)
                append_to_dset(test_dset, X_test)
                
                # 释放内存
                del batch, X_train_val, X_test, X_train, X_val
                gc.collect()
                
                if (i // batch_size) % 10 == 0:
                    print(f"正样本处理进度: {i}/{total_samples}")
        
        # 处理负样本（类似正样本处理流程）
        with h5py.File(os.path.join(PROCESSED_PATH, "one_hot_encodings.h5"), 'r') as hf:
            negative_dset = hf['negative']
            total_samples = len(negative_dset)
            
            print(f"处理负样本: {total_samples} 条记录")
            
            for i in range(0, total_samples, batch_size):
                batch = negative_dset[i:i+batch_size].astype(np.float32)
                labels = np.zeros(len(batch), dtype=int)
                
                # 分割批次数据
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    batch, labels, test_size=0.2, random_state=42, stratify=labels
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
                )
                
                # 追加到输出数据集
                append_to_dset(train_dset, X_train)
                append_to_dset(val_dset, X_val)
                append_to_dset(test_dset, X_test)
                
                # 释放内存
                del batch, X_train_val, X_test, X_train, X_val
                gc.collect()
                
                if (i // batch_size) % 10 == 0:
                    print(f"负样本处理进度: {i}/{total_samples}")
        
        # 在主HDF5文件仍处于打开状态时，将数据复制到各个子集文件
        print("正在将一热编码数据复制到各个子集文件夹...")
        for subset in ["train", "validation", "test"]:
            subset_output_path = os.path.join(OUTPUT_PATH, subset, "one_hot_encodings.h5")
            with h5py.File(subset_output_path, 'w') as subset_hf:
                # 复制整个数据集
                subset_hf.create_dataset(subset, data=out_hf[subset][:])  # [:]确保复制整个数据集
    
    print("一热编码数据处理完成")

# 处理序列数据（分批处理）
def process_sequence_data():
    print("开始处理序列数据...")
    
    # 处理正样本序列
    positive_file = os.path.join(PROCESSED_PATH, "positive_seqs.txt")
    total_positive = get_csv_row_count(positive_file)
    print(f"正样本序列总数: {total_positive}")
    
    # 创建输出文件
    train_seqs_file = os.path.join(OUTPUT_PATH, "train", "sequences.txt")
    train_labels_file = os.path.join(OUTPUT_PATH, "train", "labels.txt")
    val_seqs_file = os.path.join(OUTPUT_PATH, "validation", "sequences.txt")
    val_labels_file = os.path.join(OUTPUT_PATH, "validation", "labels.txt")
    test_seqs_file = os.path.join(OUTPUT_PATH, "test", "sequences.txt")
    test_labels_file = os.path.join(OUTPUT_PATH, "test", "labels.txt")
    
    # 清空现有文件（如果存在）
    for f in [train_seqs_file, train_labels_file, val_seqs_file, val_labels_file, test_seqs_file, test_labels_file]:
        if os.path.exists(f):
            open(f, 'w').close()
    
    # 分批处理
    batch_size = 10000
    for i in range(0, total_positive, batch_size):
        # 读取批次数据
        batch = np.loadtxt(positive_file, dtype=str, skiprows=i+1, max_rows=batch_size)
        labels = np.ones(len(batch), dtype=int)
        
        # 分割批次数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            batch, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
        )
        
        # 保存到文件
        append_to_text_file(train_seqs_file, X_train, '%s')
        append_to_text_file(train_labels_file, y_train, '%d')
        append_to_text_file(val_seqs_file, X_val, '%s')
        append_to_text_file(val_labels_file, y_val, '%d')
        append_to_text_file(test_seqs_file, X_test, '%s')
        append_to_text_file(test_labels_file, y_test, '%d')
        
        # 释放内存
        del batch, X_train_val, X_test, X_train, X_val
        gc.collect()
        
        if (i // batch_size) % 10 == 0:
            print(f"正样本序列处理进度: {i}/{total_positive}")
    
    # 处理负样本序列（类似正样本处理流程）
    negative_file = os.path.join(PROCESSED_PATH, "negative_seqs.txt")
    total_negative = get_csv_row_count(negative_file)
    print(f"负样本序列总数: {total_negative}")
    
    for i in range(0, total_negative, batch_size):
        batch = np.loadtxt(negative_file, dtype=str, skiprows=i+1, max_rows=batch_size)
        labels = np.zeros(len(batch), dtype=int)
        
        # 分割批次数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            batch, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
        )
        
        # 保存到文件
        append_to_text_file(train_seqs_file, X_train, '%s')
        append_to_text_file(train_labels_file, y_train, '%d')
        append_to_text_file(val_seqs_file, X_val, '%s')
        append_to_text_file(val_labels_file, y_val, '%d')
        append_to_text_file(test_seqs_file, X_test, '%s')
        append_to_text_file(test_labels_file, y_test, '%d')
        
        # 释放内存
        del batch, X_train_val, X_test, X_train, X_val
        gc.collect()
        
        if (i // batch_size) % 10 == 0:
            print(f"负样本序列处理进度: {i}/{total_negative}")
    
    print("序列数据处理完成")

# 处理特征数据（分批处理）
def process_feature_data():
    print("开始处理特征数据...")
    
    # 处理正样本特征
    positive_file = os.path.join(PROCESSED_PATH, "positive_features.csv")
    total_positive = get_csv_row_count(positive_file)
    print(f"正样本特征总数: {total_positive}")
    
    # 创建输出文件
    train_features_file = os.path.join(OUTPUT_PATH, "train", "features.csv")
    val_features_file = os.path.join(OUTPUT_PATH, "validation", "features.csv")
    test_features_file = os.path.join(OUTPUT_PATH, "test", "features.csv")
    
    # 清空现有文件并写入表头
    df_header = pd.read_csv(positive_file, nrows=0)
    df_header.to_csv(train_features_file, index=False)
    df_header.to_csv(val_features_file, index=False)
    df_header.to_csv(test_features_file, index=False)
    
    # 分批处理
    batch_size = 1000
    for i, chunk in enumerate(pd.read_csv(positive_file, chunksize=batch_size)):
        labels = np.ones(len(chunk), dtype=int)
        
        # 分割批次数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            chunk, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
        )
        
        # 保存到文件（不写表头）
        X_train.to_csv(train_features_file, mode='a', index=False, header=False)
        X_val.to_csv(val_features_file, mode='a', index=False, header=False)
        X_test.to_csv(test_features_file, mode='a', index=False, header=False)
        
        # 释放内存
        del chunk, X_train_val, X_test, X_train, X_val
        gc.collect()
        
        if i % 10 == 0:
            print(f"正样本特征处理进度: {i*batch_size}/{total_positive}")
    
    # 处理负样本特征（类似正样本处理流程）
    negative_file = os.path.join(PROCESSED_PATH, "negative_features.csv")
    total_negative = get_csv_row_count(negative_file)
    print(f"负样本特征总数: {total_negative}")
    
    for i, chunk in enumerate(pd.read_csv(negative_file, chunksize=batch_size)):
        labels = np.zeros(len(chunk), dtype=int)
        
        # 分割批次数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            chunk, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
        )
        
        # 保存到文件（不写表头）
        X_train.to_csv(train_features_file, mode='a', index=False, header=False)
        X_val.to_csv(val_features_file, mode='a', index=False, header=False)
        X_test.to_csv(test_features_file, mode='a', index=False, header=False)
        
        # 释放内存
        del chunk, X_train_val, X_test, X_train, X_val
        gc.collect()
        
        if i % 10 == 0:
            print(f"负样本特征处理进度: {i*batch_size}/{total_negative}")
    
    print("特征数据处理完成")

# 主函数
def main():
    try:
        # 检查输入文件是否存在
        required_files = [
            "positive_seqs.txt", "negative_seqs.txt",
            "one_hot_encodings.h5",
            "positive_features.csv", "negative_features.csv"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(PROCESSED_PATH, file)):
                raise FileNotFoundError(f"缺少必要的文件: {file}")
        
        # 处理各类数据
        process_one_hot_data()
        process_sequence_data()
        process_feature_data()
        
        # 复制原始特征文件用于参考
        shutil.copy(os.path.join(PROCESSED_PATH, "positive_features.csv"),
                    os.path.join(OUTPUT_PATH, "positive_features_original.csv"))
        shutil.copy(os.path.join(PROCESSED_PATH, "negative_features.csv"),
                    os.path.join(OUTPUT_PATH, "negative_features_original.csv"))
        
        # 统计各子集样本数量
        def count_samples(subset):
            labels_file = os.path.join(OUTPUT_PATH, subset, "labels.txt")
            if os.path.exists(labels_file):
                return sum(1 for _ in open(labels_file))
            return 0
        
        train_count = count_samples("train")
        val_count = count_samples("validation")
        test_count = count_samples("test")
        
        print(f"\n数据集分割完成！")
        print(f"训练集: {train_count} 个样本")
        print(f"验证集: {val_count} 个样本")
        print(f"测试集: {test_count} 个样本")
        print(f"结果已保存至: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()