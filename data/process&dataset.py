import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys  # 修复：添加sys模块导入


def get_desktop_path():
    """获取当前用户的桌面路径，优先使用D盘桌面"""
    try:
        # 尝试D盘桌面路径
        d_desktop = Path(r"D:\桌面")
        if d_desktop.exists():
            return d_desktop
        # 回退到系统默认桌面
        return Path(os.path.join(os.path.expanduser("~"), "Desktop"))
    except:
        # 默认返回当前工作目录
        return Path(os.getcwd())


def setup_logging(output_dir):
    """配置日志记录，确保日志文件写入指定目录"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(output_dir) / "gc_filter.log"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    return logging.getLogger(__name__)


class GenomeDataProcessor:
    """基因组数据处理工具，用于处理正样本并生成基于GC含量或长度匹配的负样本"""

    def __init__(self, config_file=None, logger=None):
        """初始化数据处理器，从配置文件或默认路径加载参数"""
        self.logger = logger or setup_logging(os.getcwd())
        self.config = self._load_config(config_file)

        # 确保输出目录存在
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"输出目录: {self.output_dir}")

        # 验证输入文件存在
        self.positive_file = self._validate_file_path(
            self.config['positive_file'], "正样本")
        self.negative_file = self._validate_file_path(
            self.config['negative_file'], "负样本")
        self.genome_file = self._validate_file_path(
            self.config['genome_file'], "基因组")

        # 加载正样本和负样本
        self.logger.info(f"加载正样本数据: {self.positive_file}")
        self.positive_samples = self._load_bed_file(self.positive_file)

        self.logger.info(f"加载负样本数据: {self.negative_file}")
        self.negative_samples = self._load_bed_file(self.negative_file)

        # 分析正样本特征
        self.positive_stats = self._analyze_positive_samples()

    def _validate_file_path(self, file_path, file_type):
        """验证文件路径是否存在，并返回绝对路径"""
        path = Path(file_path)

        # 特别处理正样本文件
        if file_type == "正样本":
            path = Path(
                r"D:\桌面\作业\genomic_data\data\raw\downloaded_data\hg38_cpg_islands.bed")

        # 检查文件是否存在
        if not path.exists():
            self.logger.error(f"{file_type}文件不存在: {path}")
            self.logger.info("请检查以下事项:")
            self.logger.info(f"  1. 文件是否实际存在于该路径")
            self.logger.info(f"  2. 文件名是否正确（包括大小写和扩展名）")
            self.logger.info(f"  3. 文件权限是否允许读取")

            # 尝试列出目录内容以帮助诊断
            try:
                parent_dir = path.parent
                if parent_dir.exists():
                    self.logger.info(f"\n{parent_dir} 目录内容:")
                    for item in parent_dir.iterdir():
                        self.logger.info(f"  - {item.name}")
                else:
                    self.logger.info(f"父目录不存在: {parent_dir}")
            except Exception as e:
                self.logger.error(f"无法列出目录内容: {e}")

            raise FileNotFoundError(f"{file_type}文件不存在: {path}")

        self.logger.info(f"找到{file_type}文件: {path}")

        # 检查文件大小是否合理
        try:
            file_size = path.stat().st_size
            if file_size < 1024:  # 小于1KB
                self.logger.warning(
                    f"{file_type}文件异常小 ({file_size} 字节)，可能为空文件")
        except Exception as e:
            self.logger.error(f"无法获取{file_type}文件大小: {e}")

        return path

    def _load_config(self, config_file):
        """加载配置文件或使用默认配置"""
        # 使用D盘路径作为默认路径
        desktop = get_desktop_path()
        default_dir = desktop / "作业" / "genomic_data" / "data"

        default_config = {
            'positive_file': str(default_dir / "raw" / "downloaded_data" / "hg38_cpg_islands.bed"),
            'negative_file': str(default_dir / "processed" / "negative_samples_both.bed"),
            'genome_file': str(default_dir / "raw" / "downloaded_data" / "hg38.fa"),
            'output_dir': str(default_dir / "processed"),
            'gc_precision': 0.1,
            'length_precision': 0.1,
            'exclude_buffer': 1000,
            'validation_output': "validation_report.txt",
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_state': 42,
            'gc_column': 'score',  # 指定GC含量列名
            'force_process': True  # 强制处理所有步骤
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            default_config[key.strip()] = value.strip()
                self.logger.info(f"已加载配置文件: {config_file}")
            except Exception as e:
                self.logger.warning(f"配置文件加载失败: {e}，使用默认配置")
        else:
            self.logger.info(
                f"使用默认配置，正样本路径: {default_config['positive_file']}")

        return default_config

    def _load_bed_file(self, bed_file):
        """加载BED格式文件，自动检测列数和格式"""
        try:
            # 读取前10行确定列数
            header = pd.read_csv(bed_file, sep='\t', nrows=10, header=None)
            n_cols = header.shape[1]

            if n_cols < 3:
                raise ValueError(f"BED文件列数不足: {n_cols} (至少需要3列)")

            # 根据列数设置列名
            if n_cols >= 6:
                col_names = ['chrom', 'start',
                             'end', 'name', 'score', 'strand']
            elif n_cols == 5:
                col_names = ['chrom', 'start', 'end', 'name', 'score']
            elif n_cols == 4:
                col_names = ['chrom', 'start', 'end', 'name']
            else:
                col_names = ['chrom', 'start', 'end']

            # 读取完整文件
            df = pd.read_csv(bed_file, sep='\t', header=None, names=col_names)
            self.logger.info(f"成功加载 {len(df)} 个BED条目")

            # 验证坐标有效性
            if not (df['end'] > df['start']).all():
                invalid = df[df['end'] <= df['start']]
                self.logger.warning(
                    f"发现 {len(invalid)} 个无效坐标条目（end ≤ start），已过滤")
                df = df[df['end'] > df['start']]

            # 检查GC含量列是否存在
            gc_column = self.config.get('gc_column', 'score')
            if gc_column not in df.columns:
                self.logger.warning(
                    f"BED文件中不存在GC含量列 '{gc_column}'，将无法计算GC统计信息")
                df['gc_content'] = np.nan  # 添加空的GC含量列
            elif gc_column != 'score':
                df['gc_content'] = df[gc_column]  # 重命名GC含量列
                self.logger.info(f"使用 '{gc_column}' 列作为GC含量")
            else:
                df['gc_content'] = df['score']  # 默认使用score列

            return df
        except Exception as e:
            self.logger.error(f"加载BED文件失败: {e}")
            raise

    def _analyze_positive_samples(self):
        """分析正样本的基本特征"""
        self.logger.info("分析正样本特征...")

        # 计算每个正样本的长度
        self.positive_samples['length'] = self.positive_samples['end'] - \
            self.positive_samples['start']

        # 计算正样本的统计信息
        stats = {
            'count': len(self.positive_samples),
            'mean_length': self.positive_samples['length'].mean(),
            'std_length': self.positive_samples['length'].std(),
            'min_length': self.positive_samples['length'].min(),
            'max_length': self.positive_samples['length'].max(),
        }

        # 检查是否有GC含量列
        if 'gc_content' in self.positive_samples.columns:
            stats['mean_gc_content'] = self.positive_samples['gc_content'].mean()
            stats['std_gc_content'] = self.positive_samples['gc_content'].std()
            stats['min_gc_content'] = self.positive_samples['gc_content'].min()
            stats['max_gc_content'] = self.positive_samples['gc_content'].max()

            # 检查GC含量是否合理
            if stats['mean_gc_content'] > 100:
                self.logger.warning(
                    f"警告: GC含量均值异常高 ({stats['mean_gc_content']:.2f}%)，可能是数据错误或列解析错误")
                self.logger.warning(
                    f"请确认BED文件的 '{self.config.get('gc_column', 'score')}' 列是否存储GC含量")
                self.logger.warning("正在尝试将GC含量标准化到0-100范围...")

                # 尝试标准化GC含量
                if (self.positive_samples['gc_content'] > 100).any():
                    self.positive_samples['gc_content'] = self.positive_samples['gc_content'] / 100
                    stats['mean_gc_content'] = self.positive_samples['gc_content'].mean()
                    self.logger.info(
                        f"GC含量已标准化，新均值: {stats['mean_gc_content']:.2f}%")

            self.logger.info(f"正样本平均GC含量: {stats['mean_gc_content']:.2f}%")
        else:
            stats['mean_gc_content'] = 0
            stats['std_gc_content'] = 0
            self.logger.warning("正样本BED文件中未找到GC含量列，无法计算GC统计信息")

        self.logger.info(
            f"正样本统计: 数量={stats['count']}, 平均长度={stats['mean_length']:.2f}bp")
        return stats

    def validate_samples(self, output_report=None):
        """验证正样本和负样本的质量"""
        self.logger.info("开始验证样本质量...")

        # 计算长度统计信息
        positive_lengths = self.positive_samples['end'] - \
            self.positive_samples['start']
        negative_lengths = self.negative_samples['end'] - \
            self.negative_samples['start']

        # 计算长度差异
        length_diff_percent = abs(positive_lengths.mean(
        ) - negative_lengths.mean()) / positive_lengths.mean() * 100

        report = {
            'positive_count': len(self.positive_samples),
            'negative_count': len(self.negative_samples),
            'positive_mean_length': positive_lengths.mean(),
            'negative_mean_length': negative_lengths.mean(),
            'length_difference_percent': length_diff_percent,
        }

        # 检查GC含量
        if 'gc_content' in self.positive_samples.columns and 'gc_content' in self.negative_samples.columns:
            positive_gc = self.positive_samples['gc_content']
            negative_gc = self.negative_samples['gc_content']

            report['positive_mean_gc'] = positive_gc.mean()
            report['negative_mean_gc'] = negative_gc.mean()
            report['gc_difference_percent'] = abs(
                positive_gc.mean() - negative_gc.mean()) / positive_gc.mean() * 100

            self.logger.info(f"正样本平均GC含量: {report['positive_mean_gc']:.2f}%")
            self.logger.info(f"负样本平均GC含量: {report['negative_mean_gc']:.2f}%")
            self.logger.info(
                f"GC含量差异百分比: {report['gc_difference_percent']:.2f}%")
        else:
            report['positive_mean_gc'] = 0
            report['negative_mean_gc'] = 0
            report['gc_difference_percent'] = 0
            self.logger.warning("无法计算GC含量差异，正样本或负样本缺少gc_content列")

        # 打印验证结果
        self.logger.info(f"正样本数量: {report['positive_count']}")
        self.logger.info(f"负样本数量: {report['negative_count']}")
        self.logger.info(f"正样本平均长度: {report['positive_mean_length']:.2f}bp")
        self.logger.info(f"负样本平均长度: {report['negative_mean_length']:.2f}bp")
        self.logger.info(
            f"长度差异百分比: {report['length_difference_percent']:.2f}%")

        # 保存验证报告
        if output_report:
            output_path = self.output_dir / output_report
            with open(output_path, 'w') as f:
                f.write("样本验证报告\n")
                f.write("=" * 50 + "\n")
                for key, value in report.items():
                    f.write(f"{key}: {value:.2f}\n" if isinstance(
                        value, float) else f"{key}: {value}\n")

            self.logger.info(f"验证报告已保存至: {output_path}")

        return report

    def merge_samples(self, output_file="merged_samples.bed"):
        """合并正样本和负样本，并添加标签列"""
        self.logger.info("开始合并正样本和负样本...")

        # 创建标签列（1=正样本，0=负样本）
        positive_labeled = self.positive_samples.copy()
        positive_labeled['label'] = 1

        negative_labeled = self.negative_samples.copy()
        negative_labeled['label'] = 0

        # 确保列名一致
        common_columns = list(set(positive_labeled.columns)
                              & set(negative_labeled.columns))
        positive_labeled = positive_labeled[common_columns]
        negative_labeled = negative_labeled[common_columns]

        # 合并数据
        merged = pd.concat(
            [positive_labeled, negative_labeled], ignore_index=True)

        # 保存合并后的文件
        output_path = self.output_dir / output_file
        merged.to_csv(output_path, sep='\t', header=False, index=False)

        self.logger.info(
            f"成功合并 {len(positive_labeled)} 个正样本和 {len(negative_labeled)} 个负样本")
        self.logger.info(f"合并后的数据保存至: {output_path}")

        return merged

    def split_dataset(self, merged_data, output_prefix="dataset"):
        """将合并后的数据划分为训练集、验证集和测试集"""
        self.logger.info("开始划分数据集...")

        # 获取配置参数
        train_ratio = float(self.config['train_ratio'])
        val_ratio = float(self.config['val_ratio'])
        test_ratio = float(self.config['test_ratio'])
        random_state = int(self.config['random_state'])

        # 验证比例之和是否为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例之和必须为1，但得到 {total_ratio}")

        # 按标签分层抽样
        positive_data = merged_data[merged_data['label'] == 1]
        negative_data = merged_data[merged_data['label'] == 0]

        # 划分正样本
        train_pos, val_pos, test_pos = self._split_data(
            positive_data, train_ratio, val_ratio, random_state)

        # 划分负样本
        train_neg, val_neg, test_neg = self._split_data(
            negative_data, train_ratio, val_ratio, random_state)

        # 合并正样本和负样本
        train_dataset = pd.concat([train_pos, train_neg], ignore_index=True)
        val_dataset = pd.concat([val_pos, val_neg], ignore_index=True)
        test_dataset = pd.concat([test_pos, test_neg], ignore_index=True)

        # 保存数据集
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        for suffix, dataset in datasets.items():
            output_file = self.output_dir / f"{output_prefix}_{suffix}.bed"
            dataset.to_csv(output_file, sep='\t', header=False, index=False)
            self.logger.info(
                f"{suffix}集包含 {len(dataset)} 个样本，已保存至: {output_file}")

        # 生成数据集统计报告
        stats = {
            'total_samples': len(merged_data),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'train_positive': len(train_pos),
            'train_negative': len(train_neg),
            'val_positive': len(val_pos),
            'val_negative': len(val_neg),
            'test_positive': len(test_pos),
            'test_negative': len(test_neg)
        }

        return stats

    def _split_data(self, data, train_ratio, val_ratio, random_state):
        """将数据按比例划分为训练集、验证集和测试集"""
        from sklearn.model_selection import train_test_split

        # 先划分训练集和临时集
        train, temp = train_test_split(
            data,
            train_size=train_ratio,
            random_state=random_state,
            shuffle=True
        )

        # 再将临时集划分为验证集和测试集
        val_ratio_remaining = val_ratio / (1 - train_ratio)
        val, test = train_test_split(
            temp,
            train_size=val_ratio_remaining,
            random_state=random_state,
            shuffle=True
        )

        return train, val, test


def main():
    """主函数，处理命令行参数并执行数据处理任务"""
    parser = argparse.ArgumentParser(description='处理基因组数据并生成训练数据集')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--validate', action='store_true', help='执行样本验证')
    parser.add_argument('--merge', action='store_true', help='合并正样本和负样本')
    parser.add_argument('--split', action='store_true', help='划分训练集、验证集和测试集')
    args = parser.parse_args()

    try:
        # 使用D盘路径作为默认输出目录
        desktop = get_desktop_path()
        default_output = desktop / "作业" / "genomic_data" / "data" / "processed"
        default_output.mkdir(exist_ok=True, parents=True)

        # 初始化日志
        logger = setup_logging(default_output)
        logger.info("=" * 50)
        logger.info("开始处理基因组数据...")
        logger.info(f"Python版本: {sys.version}")  # 修复：sys已导入
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info("=" * 50)

        # 初始化数据处理器
        processor = GenomeDataProcessor(args.config, logger)

        # 执行验证
        validation_output = processor.config.get(
            'validation_output', 'validation_report.txt')
        processor.validate_samples(output_report=validation_output)
        logger.info(f"验证报告已生成: {processor.output_dir / validation_output}")

        # 执行合并
        merged = processor.merge_samples()
        logger.info(
            f"合并后的样本已保存: {processor.output_dir / 'merged_samples.bed'}")

        # 执行划分
        stats = processor.split_dataset(merged)
        logger.info("数据集划分完成:")
        logger.info(f"  - 训练集: {stats['train_samples']} 样本")
        logger.info(f"  - 验证集: {stats['val_samples']} 样本")
        logger.info(f"  - 测试集: {stats['test_samples']} 样本")

        # 生成处理完成报告
        report_file = processor.output_dir / "processing_complete.txt"
        with open(report_file, 'w') as f:
            f.write("数据处理任务全部完成!\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n")
            f.write(f"输出目录: {processor.output_dir}\n\n")
            f.write("生成的文件列表:\n")
            f.write(f" 1. 验证报告: {validation_output}\n")
            f.write(f" 2. 合并样本: merged_samples.bed\n")
            f.write(f" 3. 训练集: dataset_train.bed\n")
            f.write(f" 4. 验证集: dataset_val.bed\n")
            f.write(f" 5. 测试集: dataset_test.bed\n")
            f.write(f" 6. 日志文件: gc_filter.log\n")

        logger.info(f"处理完成报告已保存至: {report_file}")
        logger.info(f"所有输出文件位于: {processor.output_dir}")
        logger.info("=" * 50)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"处理过程中发生错误: {e}")
        try:
            error_log = Path(
                r"D:\桌面\作业\genomic_data\data\processed\genomic_data_error.log")
            with open(error_log, 'w') as f:
                f.write(f"程序运行错误: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
            logger.error(f"错误详情已保存至: {error_log}")
        except:
            logger.error("无法保存详细错误日志")
        exit(1)


if __name__ == "__main__":
    main()
