import os
import wget

# 数据存储基础路径
base_path = r"D:\桌面\作业\genomic_data\data\raw"
# 创建存储目录（如果不存在）
if not os.path.exists(base_path):
    os.makedirs(base_path)

# 2. 下载参考基因组序列（人类 hg38 ，这里以 UCSC 数据库的全基因组序列为例，可根据需求调整）
# 人类 hg38 参考基因组序列（FASTA 格式，全基因组）的 UCSC 下载链接示例
genome_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
genome_file = os.path.join(base_path, "hg38.fa.gz")
if not os.path.exists(genome_file):
    print("开始下载参考基因组序列（hg38）...")
    wget.download(genome_url, genome_file)
    print("\n参考基因组序列下载完成")
else:
    print("参考基因组序列已存在，无需重复下载")

# 3. 下载对应的 CpG 岛注释文件（BED 格式，从 UCSC 数据库获取）
# UCSC 中人类 hg38 的 CpG 岛注释文件（BED 格式）下载链接示例
cpg_bed_url = "http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz"
# 这里将下载的压缩文件解压后命名为 cpgIslandExt.bed（实际可根据需求处理），先下载压缩包
cpg_bed_gz_file = os.path.join(base_path, "cpgIslandExt.txt.gz")
if not os.path.exists(cpg_bed_gz_file):
    print("开始下载 CpG 岛注释文件（BED 格式）...")
    wget.download(cpg_bed_url, cpg_bed_gz_file)
    print("\nCpG 岛注释文件（压缩包）下载完成，如需解压可进一步处理")
else:
    print("CpG 岛注释文件（压缩包）已存在，无需重复下载")
