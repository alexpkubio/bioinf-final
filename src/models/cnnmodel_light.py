#导入pythorch及相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

#定义CNN模型
class CpGisland_CNN(nn.Module):
    def __init__(self, seq_length=1200, num_channels=4, num_filters=(16, 32, 64),kernel_sizes=(3, 5, 7), dropout_rate=0.6, original_feature_dim=0):  # 增加Dropout比率
        super(CpGisland_CNN, self).__init__()
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.original_feature_dim = original_feature_dim

        # 序列处理分支初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_filters[0], kernel_size=kernel_sizes[0], padding=1),
            nn.BatchNorm1d(num_filters[0]),  # 使用批归一化
            nn.ReLU()
        )

        # 残差块增强特征提取能力
        self.res_blocks = nn.ModuleList()
        for i in range(1, len(num_filters)):
            self.res_blocks.append(
                ResidualBlock(num_filters[i - 1], num_filters[i], kernel_sizes[i % len(kernel_sizes)])
            )

        # 注意力层
        self.attention = Selfattention(num_filters[-1])

        # 全局池化层
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 序列特征维度
        self.seq_feature_dim = num_filters[-1] * 2

        # 原始特征处理
        if original_feature_dim > 0:
            self.original_feature_fc = nn.Sequential(
                nn.Linear(original_feature_dim, 32),  # 全连接层
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2) # 防止过拟合
            )

        #融合层使用原始特征增加维度
        fusion_input_dim = self.seq_feature_dim
        if original_feature_dim > 0:
            fusion_input_dim += 32

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # 合理控制在0.5
            nn.Linear(64, 1)
        )

    def forward(self, seq_input, original_features, label):
        # 处理序列数据
        x = self.initial_conv(seq_input)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.attention(x)
        max_pooled = self.global_max_pool(x).squeeze(-1)
        avg_pooled = self.global_avg_pool(x).squeeze(-1)
        seq_features = torch.cat([max_pooled, avg_pooled], dim=1)

        # 处理原始特征
        if self.original_feature_dim > 0 and original_features is not None:
            orig_features = self.original_feature_fc(original_features)
        else:
            orig_features = None

        # 拼接序列特征和原始特征
        if orig_features is not None:
            combined = torch.cat([seq_features, orig_features], dim=1)
        else:  #写这句是考虑到适用性，我们的代码第一次运行是并没有考虑原始特征
            combined = seq_features

        output = self.fc(combined)
        return output.squeeze()

# 残差块类，上面的CNN模型需要
class ResidualBlock(nn.Module):
    def __init__(self, inpt_channels, outpt_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        #第一个卷积层
        self.conv1 = nn.Conv1d(inpt_channels, outpt_channels, kernel_size, padding=kernel_size // 2, stride=2)
        self.bn1 = nn.BatchNorm1d(outpt_channels)

        #第二个卷积层
        self.conv2 = nn.Conv1d(outpt_channels, outpt_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(outpt_channels)

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv1d(inpt_channels, outpt_channels, kernel_size=1, stride=2),
            nn.BatchNorm1d(outpt_channels)
        )

    def forward(self, x):
        # 替换csv文件中的NaN为0
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # 卷积+批归一化,多层形式，并进行残差连接
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

# 自注意力机制类 - 增加数值稳定性
class Selfattention(nn.Module):
    def __init__(self, in_channels):
        super(Selfattention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = 1e-12  # 添加小值防止除零

    def forward(self, x):
        # 替换NaN为0，这是因为原始csv中部分空格为nan，其实就是无数据，认为是0
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        batch_size, C, width = x.size()
        cal_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
        cal_key = self.key(x).view(batch_size, -1, width)
        cal_value = self.value(x).view(batch_size, -1, width)

        # 计算注意力权重
        energy = torch.bmm(cal_query, cal_key)
        energy_scaled = energy / (cal_key.size(-2) ** 0.5 + self.eps)

        # 使用带温度的softmax注意力权重
        attention = F.softmax(energy_scaled, dim=-1)

        # 检查注意力权重是否包含NaN，如果有nan，尝试均匀注意力
        if torch.isnan(attention).any():
            attention = torch.ones_like(energy_scaled) / energy_scaled.size(-1)

        #输出
        outpt = torch.bmm(cal_value, attention.permute(0, 2, 1))
        outpt = outpt.view(batch_size, C, width)
        outpt = self.gamma * outpt + x
        return outpt