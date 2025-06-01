import torch
import torch.nn as nn
import torch.nn.functional as F

class CpGisland_CNN(nn.Module):
	def __init__(self,seq_lenth=1200, num_channels=4, num_filters=(32, 64, 128, 256), kernel_sizes=(3, 5, 7, 9), dropout_rate=0.3):#由于CpG岛的长度大致在300-2000bp，选用一个中间值1200
		super(CpGisland_CNN, self).__init__()
		self.seq_length = seq_lenth
		self.num_channels = num_channels
		self.num_filters = num_filters
		self.initial_conv = nn.Sequential(
			nn.Conv1d(in_channels=num_channels, out_channels=num_filters[0],kernel_size=kernel_sizes[0], padding=1),
			nn.BatchNorm1d(num_filters[0]),
			nn.ReLU()
		)# 初始卷积层搭建

		# 残差块
		self.res_blocks = nn.ModuleList()
		for i in range(1, len(num_filters)):
			self.res_blocks.append(
				ResidualBlock(num_filters[i - 1], num_filters[i], kernel_sizes[i % len(kernel_sizes)]) #便于num_filters 和kernel_size不同时
			)

		# 注意力层
		self.attention = Selfattention(num_filters[-1])

		self.featuresize=seq_lenth
		for _ in range(len(num_filters)-1):
			self.featuresize //= 2

		self.global_max_pool = nn.AdaptiveMaxPool1d(1)
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

		self.fc1 = nn.Linear(num_filters[-1] * 2, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 1)
		self.dropout = nn.Dropout(dropout_rate)

		self.dropout = nn.Dropout(dropout_rate)
	def forward(self, x):
		x = self.initial_conv(x)
		x = x.permute(0, 2, 1)

		x = self.initial_conv(x)#初始卷积

		for res_block in self.res_blocks:#残差块
			x = res_block(x)

		x = self.attention(x) #注意力机制

		max_pooled = self.global_max_pool(x).squeeze(-1)#池化
		avg_pooled = self.global_avg_pool(x).squeeze(-1)
		x = torch.cat([max_pooled, avg_pooled], dim=1)

		x = F.relu(self.fc1(x))#连接层
		x = self.dropout(x)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)


		return torch.sigmoid(x).squeeze()

class ResidualBlock(nn.Module):#残差块机制的引入，
	def __init__(self, inpt_channels, outpt_channels, kernel_size=3):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv1d(inpt_channels, outpt_channels, kernel_size,padding=kernel_size//2, stride=2)
		self.bn1 = nn.BatchNorm1d(outpt_channels)
		self.conv2 = nn.Conv1d(outpt_channels, outpt_channels, kernel_size,padding=kernel_size//2)
		self.bn2 = nn.BatchNorm1d(outpt_channels)
		self.shortcut = nn.Sequential(
		nn.Conv1d(inpt_channels, outpt_channels, kernel_size=1, stride=2),
		nn.BatchNorm1d(outpt_channels)
		)
	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(residual)
		out = F.relu(out)
		return out

class Selfattention(nn.Module):#引入attention机制
	def __init__(self, in_channels):
		super(Selfattention, self).__init__()
		self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
		self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
		self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))

		def forward(self, x):
			batch_size, C, width = x.size()
			cal_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
			cal_key = self.key(x).view(batch_size, -1, width)
			cal_value = self.value(x).view(batch_size, -1, width)
			energy = torch.bmm(cal_query, cal_key)
			attention = F.softmax(energy, dim=-1)
			outpt = torch.bmm(cal_value, attention.permute(0, 2, 1))
			outpt = outpt.view(batch_size, C, width)
			outpt = self.gamma * outpt + x
			return outpt
