import os
import pandas as pd
import numpy as np
import random
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenomeDataProcessor:
    """基因组数据处理工具，流式处理基因组数据以减少内存消耗"""

    def __init__(self, genome_file, positive_bed_file, output_dir="processed_data"):
        """初始化数据处理器"""
        self.genome_file = Path(genome_file)
        self.positive_bed_file = Path(positive_bed_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 不加载整个基因组到内存，仅记录染色体名称和长度
        self.chrom_lengths = self._get_chrom_lengths(self.genome_file)

        # 加载正样本数据
        logger.info(f"加载正样本数据: {self.positive_bed_file}")
        self.positive_samples = self._load_bed_file(self.positive_bed_file)

        # 统计正样本的基本特征（无需加载序列，仅用BED文件中的长度）
        self.positive_stats = self._analyze_positive_samples()

    def _get_chrom_lengths(self, genome_file):
        """获取各染色体长度（流式读取，不加载序列）"""
        chrom_lengths = {}
        current_chrom = None

        try:
            with open(genome_file, 'r') as f:
                for line in tqdm(f, desc="解析染色体长度"):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('>'):
                        # 提取染色体名称（去除注释部分）
                        chrom_name = line[1:].split()[0]
                        current_chrom = chrom_name
                        chrom_lengths[current_chrom] = 0
                    else:
                        if current_chrom is not None:
                            chrom_lengths[current_chrom] += len(line.strip())

            logger.info(f"成功获取 {len(chrom_lengths)} 条染色体长度")
            return chrom_lengths

        except Exception as e:
            logger.error(f"解析染色体长度失败: {e}")
            raise

    def _load_bed_file(self, bed_file):
        """加载BED格式文件"""
        try:
            df = pd.read_csv(bed_file, sep='\t', header=None)
            if len(df.columns) < 6:
                raise ValueError(f"BED文件列数不足: {len(df.columns)}")

            df.columns = ['chrom', 'start', 'end', 'name', 'score', 'strand']
            logger.info(f"成功加载 {len(df)} 个BED条目")
            return df
        except Exception as e:
            logger.error(f"加载BED文件失败: {e}")
            raise

    def _analyze_positive_samples(self):
        """分析正样本特征（仅使用BED文件中的长度，无需序列）"""
        logger.info("分析正样本特征...")
        self.positive_samples['length'] = self.positive_samples['end'] - \
            self.positive_samples['start']

        mean_length = self.positive_samples['length'].mean()
        std_length = self.positive_samples['length'].std()
        min_length = self.positive_samples['length'].min()
        max_length = self.positive_samples['length'].max()

        return {
            'count': len(self.positive_samples),
            'mean_length': mean_length,
            'std_length': std_length,
            'min_length': min_length,
            'max_length': max_length,
            'mean_gc_content': 0,  # 移除GC含量计算，避免内存消耗
            'std_gc_content': 0
        }

    def generate_negative_samples(self, strategy='both', match_precision=0.1,
                                  max_attempts=100, exclude_buffer=1000,
                                  chromosomes=None, output_file=None):
        """生成负样本（仅依赖染色体长度，不加载序列）"""
        logger.info(f"开始生成负样本，匹配策略: {strategy}")
        if 'gc' in strategy:
            logger.warning("警告：GC含量匹配已禁用（内存不足），自动切换为长度匹配")
            strategy = 'length'  # 强制使用长度匹配

        # 确定输出文件名
        if not output_file:
            output_file = self.output_dir / f"negative_samples_{strategy}.bed"

        # 筛选染色体
        valid_chromosomes = self._get_valid_chromosomes(chromosomes)

        # 构建排除区域（仅记录染色体和长度，不存储具体区间）
        exclude_regions = self._build_exclude_regions(
            valid_chromosomes, exclude_buffer)

        # 长度匹配参数
        length_min = self.positive_stats['mean_length'] * (1 - match_precision)
        length_max = self.positive_stats['mean_length'] * (1 + match_precision)
        logger.info(f"长度匹配阈值: {length_min:.0f}-{length_max:.0f}bp")

        # 生成负样本（仅使用染色体长度，不读取序列）
        negative_samples = []
        progress_bar = tqdm(total=len(self.positive_samples), desc="生成负样本")

        while len(negative_samples) < len(self.positive_samples):
            template = self.positive_samples.sample(1).squeeze()
            target_length = template['length']
            success, sample = self._generate_length_matched_sample(
                valid_chromosomes, exclude_regions, target_length,
                length_min, length_max, max_attempts
            )

            if success:
                negative_samples.append(sample)
                progress_bar.update(1)

        progress_bar.close()
        self._save_samples(negative_samples, output_file)
        logger.info(f"负样本生成完成: {output_file}")
        return output_file

    def _get_valid_chromosomes(self, chromosomes=None):
        """获取有效染色体列表"""
        if chromosomes:
            valid = [c for c in chromosomes if c in self.chrom_lengths]
            if not valid:
                raise ValueError(f"指定染色体均不存在: {chromosomes}")
            return valid
        else:
            # 过滤出标准染色体（如chr1-22,chrX,chrY）
            return [c for c in self.chrom_lengths if any(c.startswith(p) for p in ['chr1', 'chr2', 'chrX'])]

    def _build_exclude_regions(self, chromosomes, buffer_size):
        """构建排除区域（仅记录染色体和总长度，不存储具体区间）"""
        return {chrom: self.chrom_lengths[chrom] for chrom in chromosomes}

    def _generate_length_matched_sample(self, chromosomes, exclude_lengths,
                                        target_length, length_min, length_max, max_attempts):
        """生成长度匹配的负样本（无需序列数据）"""
        for _ in range(max_attempts):
            chrom = random.choice(chromosomes)
            chrom_len = exclude_lengths[chrom]
            length = np.random.uniform(length_min, length_max)
            length = max(100, min(int(length), chrom_len - 100))  # 最小长度100bp

            start = np.random.randint(0, chrom_len - length)
            end = start + length

            return True, [chrom, start, end, "negative", 0, "+"]  # 虚拟分数

    def _save_samples(self, samples, output_file):
        """保存样本到BED文件"""
        df = pd.DataFrame(samples, columns=[
                          'chrom', 'start', 'end', 'name', 'score', 'strand'])
        df.to_csv(output_file, sep='\t', header=False, index=False)
        logger.info(f"保存 {len(samples)} 个样本到 {output_file}")


def main():
    """主函数"""
    default_positive_file = r"D:\桌面\作业\genomic_data\data\raw\downloaded_data\hg38_cpg_islands.bed"
    default_genome_file = r"D:\桌面\作业\genomic_data\data\raw\downloaded_data\hg38.fa"
    default_output_dir = r"D:\桌面\作业\genomic_data\data\processed"

    parser = argparse.ArgumentParser()
    parser.add_argument('--genome_file', default=default_genome_file)
    parser.add_argument('--positive_bed_file', default=default_positive_file)
    parser.add_argument('--output_dir', default=default_output_dir)
    args = parser.parse_args()

    try:
        processor = GenomeDataProcessor(
            genome_file=args.genome_file,
            positive_bed_file=args.positive_bed_file,
            output_dir=args.output_dir
        )
        processor.generate_negative_samples()
        logger.info("任务完成")
    except Exception as e:
        logger.error(f"失败: {e}")


if __name__ == "__main__":
    main()
