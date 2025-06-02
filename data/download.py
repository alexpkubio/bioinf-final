import os
import requests
import gzip
import shutil
from tqdm import tqdm
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenomeDataDownloader:
    """基因组数据下载工具，支持下载基因组序列和CpG岛注释"""

    def __init__(self, output_dir="downloaded_data"):
        """初始化下载器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 预定义常用基因组下载URL模板
        self.genome_url_templates = {
            "hg38": {
                "fasta": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
                "cpg_bed": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
            },
            "hg19": {
                "fasta": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
                "cpg_bed": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cpgIslandExt.txt.gz"
            },
            "mm10": {
                "fasta": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
                "cpg_bed": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/cpgIslandExt.txt.gz"
            }
        }

    def download_file(self, url, output_path, chunk_size=1024):
        """下载文件并显示进度条"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # 获取文件大小用于进度显示
            total_size = int(response.headers.get('content-length', 0))

            logger.info(f"开始下载: {url}")
            logger.info(f"文件大小: {total_size / (1024*1024):.2f} MB")

            with open(output_path, 'wb') as file, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

            logger.info(f"下载完成: {output_path}")
            return output_path
        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败: {e}")
            raise

    def decompress_gzip(self, input_path, output_path):
        """解压gzip文件"""
        try:
            logger.info(f"开始解压: {input_path}")
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"解压完成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"解压失败: {e}")
            raise

    def download_genome_sequence(self, species="hg38"):
        """下载基因组序列文件"""
        if species not in self.genome_url_templates:
            raise ValueError(
                f"不支持的物种: {species}。可用选项: {list(self.genome_url_templates.keys())}")

        url = self.genome_url_templates[species]["fasta"]
        compressed_file = self.output_dir / f"{species}.fa.gz"
        decompressed_file = self.output_dir / f"{species}.fa"

        # 检查文件是否已下载
        if not compressed_file.exists():
            self.download_file(url, compressed_file)

        # 检查文件是否已解压
        if not decompressed_file.exists():
            self.decompress_gzip(compressed_file, decompressed_file)

        return decompressed_file

    def download_cpg_islands(self, species="hg38"):
        """下载CpG岛注释数据"""
        if species not in self.genome_url_templates:
            raise ValueError(
                f"不支持的物种: {species}。可用选项: {list(self.genome_url_templates.keys())}")

        url = self.genome_url_templates[species]["cpg_bed"]
        compressed_file = self.output_dir / f"{species}_cpg_islands.txt.gz"
        decompressed_file = self.output_dir / f"{species}_cpg_islands.bed"

        # 检查文件是否已下载
        if not compressed_file.exists():
            self.download_file(url, compressed_file)

        # 检查文件是否已解压并转换格式
        if not decompressed_file.exists():
            self.decompress_gzip(compressed_file, decompressed_file)
            self._convert_to_bed_format(decompressed_file)

        return decompressed_file

    def _convert_to_bed_format(self, input_file):
        """将UCSC格式的CpG岛数据转换为BED格式"""
        try:
            logger.info(f"正在转换为BED格式: {input_file}")
            with open(input_file, 'r') as f:
                lines = f.readlines()

            # UCSC CpG岛文件格式: bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum, perCpg, perGc, obsExp
            # 转换为BED格式: chrom, chromStart, chromEnd, name, score, strand
            bed_lines = []
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split('\t')
                if len(parts) >= 11:
                    chrom = parts[1]
                    start = int(parts[2])
                    end = int(parts[3])
                    name = parts[4]
                    # 使用CG含量作为分数
                    score = int(float(parts[8]) * 100)  # perCpg转换为0-100的分数
                    strand = '+'

                    bed_lines.append(
                        f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

            # 覆盖原文件
            with open(input_file, 'w') as f:
                f.writelines(bed_lines)

            logger.info(f"BED格式转换完成: {input_file}")
        except Exception as e:
            logger.error(f"格式转换失败: {e}")
            raise


def main():
    """主函数，处理命令行参数并执行下载任务"""
    parser = argparse.ArgumentParser(description='下载基因组数据和CpG岛注释')
    parser.add_argument('--species', default='hg38', choices=['hg38', 'hg19', 'mm10'],
                        help='要下载的物种基因组版本')
    parser.add_argument('--output_dir', default='downloaded_data',
                        help='下载数据的保存目录')
    args = parser.parse_args()

    try:
        logger.info(f"开始下载{args.species}基因组数据...")
        downloader = GenomeDataDownloader(args.output_dir)

        # 下载基因组序列
        genome_file = downloader.download_genome_sequence(args.species)
        logger.info(f"基因组序列已下载至: {genome_file}")

        # 下载CpG岛注释
        cpg_file = downloader.download_cpg_islands(args.species)
        logger.info(f"CpG岛注释已下载至: {cpg_file}")

        logger.info("数据下载完成!")

        # 显示下载的文件信息
        print("\n========== 下载完成 ==========")
        print(f"基因组序列文件: {genome_file}")
        print(f"CPG岛注释文件: {cpg_file}")
        print("=============================")

    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
