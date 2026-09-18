"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import torch


class NSELoss(torch.nn.Module):
    """计算（批量）NSE损失函数

    每个样本i的权重为 1 / (std_i + eps)^2，其中std_i是样本所属流域的径流标准差，
    用于考虑不同流域的特性差异。

    参数:
    -----------
    eps : float
        常数，添加到权重中用于数值稳定性和平滑处理，默认为0.1
    """

    def __init__(self, eps: float = 0.1):
        # 调用父类构造函数
        super(NSELoss, self).__init__()
        # 设置平滑常数，防止除以零
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """计算批量NSE损失函数

        参数
        ----------
        y_pred : torch.Tensor
            包含网络预测值的张量
        y_true : torch.Tensor
            包含真实径流值的张量
        q_stds : torch.Tensor
            包含每个样本的径流标准差（基于训练期计算）的张量

        返回
        -------
        torch.Tensor
            （批量）NSE损失值

        详细说明:
        ---------
        NSE（Nash-Sutcliffe效率系数）是水文模型中常用的评估指标，定义为：
        NSE = 1 - [Σ(y_true - y_pred)²] / [Σ(y_true - mean(y_true))²]
        
        在损失函数中，我们使用加权的平方误差，权重基于每个流域的径流变异性。
        这种加权方式使得模型在训练时：
        - 对径流变异性大的流域（std_i大）给予较小的权重
        - 对径流变异性小的流域（std_i小）给予较大的权重
        
        这样设计的原因是：
        1. 防止模型过度关注高流量事件
        2. 使模型在不同流量级别的预测上都能表现良好
        3. 考虑不同流域的水文响应特性差异
        """
        # 计算平方误差：(预测值 - 真实值)^2
        squared_error = (y_pred - y_true)**2
        
        # 计算权重：1 / (径流标准差 + eps)^2
        # 这样设计的原因：
        # - 径流变异性大的流域（std_i大）权重较小
        # - 径流变异性小的流域（std_i小）权重较大  
        # - eps防止分母为零，增加数值稳定性
        weights = 1 / (q_stds + self.eps)**2
        
        # 应用权重：加权平方误差
        scaled_loss = weights * squared_error

        # 返回加权损失的平均值
        return torch.mean(scaled_loss)