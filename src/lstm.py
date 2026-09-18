from typing import Tuple
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """标准LSTM的实现

    TODO: 包含论文引用和LaTeX公式

    参数
    ----------
    input_size : int
        输入特征的数量
    hidden_size : int
        隐藏层/记忆单元的数量
    batch_first : bool, 可选
        如果为True，期望批量输入的形状为[batch, seq, features]，否则形状必须为[seq, batch, features]，默认为True
    initial_forget_bias : int, 可选
        初始遗忘门偏置的值，默认为0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        # 调用父类构造函数
        super(LSTM, self).__init__()

        # 存储网络参数
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size  # 隐藏层维度
        self.batch_first = batch_first  # 批量维度是否在前
        self.initial_forget_bias = initial_forget_bias  # 初始遗忘门偏置

        # 创建可学习参数的张量
        # 输入到隐藏层的权重矩阵，形状为[input_size, 4 * hidden_size]
        # 4代表四个门：遗忘门、输入门、输出门和候选记忆单元
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        
        # 隐藏层到隐藏层的权重矩阵，形状为[hidden_size, 4 * hidden_size]
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        
        # 偏置项，形状为[4 * hidden_size]
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化LSTM的所有可学习参数"""
        # 使用正交初始化初始化输入到隐藏层的权重
        nn.init.orthogonal_(self.weight_ih.data)

        # 创建单位矩阵作为隐藏层到隐藏层权重的初始值
        weight_hh_data = torch.eye(self.hidden_size)  # 创建单位矩阵
        weight_hh_data = weight_hh_data.repeat(1, 4)  # 在列方向重复4次，对应4个门
        self.weight_hh.data = weight_hh_data  # 赋值给权重矩阵

        # 将偏置初始化为0
        nn.init.constant_(self.bias.data, val=0)

        # 如果指定了初始遗忘门偏置，则设置遗忘门的偏置
        if self.initial_forget_bias != 0:
            # 偏置的前hidden_size个元素对应遗忘门
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        参数
        ----------
        x : torch.Tensor
            包含输入序列批量的张量。格式必须匹配batch_first参数定义的格式。

        返回
        -------
        h_n : torch.Tensor
            批量中每个样本每个时间步的隐藏状态
        c_n : torch.Tensor
            批量中每个样本每个时间步的细胞状态
        """
        # 如果batch_first为True，调整张量维度为[seq_len, batch_size, features]
        if self.batch_first:
            x = x.transpose(0, 1)

        # 获取序列长度、批量大小和特征维度
        seq_len, batch_size, _ = x.size()

        # 初始化初始隐藏状态和细胞状态为0
        h_0 = x.data.new(batch_size, self.hidden_size).zero_()  # 初始隐藏状态
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()  # 初始细胞状态
        h_x = (h_0, c_0)  # 组合成元组

        # 创建空列表临时存储所有中间隐藏状态/细胞状态
        h_n, c_n = [], []

        # 将偏置向量扩展到批量大小
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # 对输入序列执行前向步骤
        for t in range(seq_len):
            # 获取当前时间步的隐藏状态和细胞状态
            h_0, c_0 = h_x

            # 计算门控值
            # 公式：gates = bias + h_0 * W_hh + x[t] * W_ih
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))
            
            # 将门控值分割为四个部分：遗忘门(f)、输入门(i)、输出门(o)、候选记忆单元(g)
            f, i, o, g = gates.chunk(4, 1)

            # LSTM核心计算：
            # 细胞状态更新：c_1 = sigmoid(f) * c_0 + sigmoid(i) * tanh(g)
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            
            # 隐藏状态更新：h_1 = sigmoid(o) * tanh(c_1)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # 将中间隐藏状态/细胞状态存储在列表中
            h_n.append(h_1)
            c_n.append(c_1)

            # 更新隐藏状态和细胞状态为下一时间步准备
            h_x = (h_1, c_1)

        # 将列表中的状态堆叠成张量
        h_n = torch.stack(h_n, 0)  # 形状: [seq_len, batch_size, hidden_size]
        c_n = torch.stack(c_n, 0)  # 形状: [seq_len, batch_size, hidden_size]

        # 如果batch_first为True，调整维度回到[batch_size, seq_len, hidden_size]
        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        # 返回所有时间步的隐藏状态和细胞状态
        return h_n, c_n