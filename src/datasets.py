from pathlib import PosixPath
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# 从datautils模块导入相关的数据处理函数
from .datautils import (load_attributes, load_discharge, load_forcing,
                        normalize_features, reshape_data)


class CamelsTXT(Dataset):
    """用于处理CAMELS数据集中原始文本文件的PyTorch数据集类。
       
    参数
    ----------
    camels_root : PosixPath
        CAMELS数据集主目录的路径
    basin : str
        8位USGS测站ID
    dates : List
        时间段的开始和结束日期
    is_train : bool
        如果为True，则对径流观测值进行归一化并移除无效的径流样本
    seq_length : int, 可选
        输入序列的长度，默认为270
    with_attributes : bool, 可选
        如果为True，加载并返回额外的流域属性，默认为False
    attribute_means : pd.Series, 可选
        流域特征的均值，用于推理时的归一化，默认为None
    attribute_stds : pd.Series, 可选
        流域特征的标准差，用于推理时的归一化，默认为None
    concat_static : bool, 可选
        如果为True，在每个时间步将流域特征添加到气象强迫输入数据中，默认为False
    db_path : str, 可选
        包含流域特征的sqlite3数据库文件的路径，默认为None
    """

    def __init__(self,
                 camels_root: PosixPath,
                 basin: str,
                 dates: List,
                 is_train: bool,
                 seq_length: int = 270,
                 with_attributes: bool = False,
                 attribute_means: pd.Series = None,
                 attribute_stds: pd.Series = None,
                 concat_static: bool = False,
                 db_path: str = None):
        # 初始化所有参数
        self.camels_root = camels_root
        self.basin = basin
        self.seq_length = seq_length
        self.is_train = is_train
        self.dates = dates
        self.with_attributes = with_attributes
        self.attribute_means = attribute_means
        self.attribute_stds = attribute_stds
        self.concat_static = concat_static
        self.db_path = db_path

        # 用于存储径流标准差的占位符，用于训练期间重新缩放损失
        self.q_std = None

        # 用于存储整个时间段（包括预热期）的开始和结束日期的占位符
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        # 加载输入和输出数据
        self.x, self.y = self._load_data()

        # 如果需要加载流域属性
        if self.with_attributes:
            self.attributes = self._load_attributes()

        # 确定样本数量
        self.num_samples = self.x.shape[0]

    def __len__(self):
        """返回数据集中的样本数量"""
        return self.num_samples

    def __getitem__(self, idx: int):
        """根据索引获取单个样本
        
        返回格式根据是否使用流域属性以及如何结合静态特征而变化
        """
        if self.with_attributes:
            if self.concat_static:
                # 如果使用静态特征连接，将流域属性重复到每个时间步并与动态特征连接
                x = torch.cat([self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1)
                return x, self.y[idx]
            else:
                # 否则，分别返回动态特征、静态特征和目标值
                return self.x[idx], self.attributes, self.y[idx]
        else:
            # 不使用流域属性，只返回动态特征和目标值
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从文本文件加载输入和输出数据"""
        # 加载气象强迫数据和流域面积
        df, area = load_forcing(self.camels_root, self.basin)
        # 加载径流数据并添加到DataFrame中
        df['QObs(mm/d)'] = load_discharge(self.camels_root, self.basin, area)

        # 我们使用(seq_length)个时间步之前的数据作为预热期
        start_date = self.dates[0] - pd.DateOffset(days=self.seq_length - 1)
        end_date = self.dates[1]
        df = df[start_date:end_date]

        # 存储所选时间段（包括预热期）的第一个和最后一个日期
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # 使用所有气象变量作为输入
        x = np.array([
            df['PRCP(mm/day)'].values,  # 降水
            df['SRAD(W/m2)'].values,    # 太阳辐射
            df['Tmax(C)'].values,       # 最高温度
            df['Tmin(C)'].values,       # 最低温度
            df['Vp(Pa)'].values         # 水汽压
        ]).T  # 转置，使每行是一个时间步，每列是一个特征

        y = np.array([df['QObs(mm/d)'].values]).T  # 径流目标值

        # 对数据进行归一化，为LSTM训练重塑形状，并移除无效样本
        x = normalize_features(x, variable='inputs')  # 归一化输入特征

        x, y = reshape_data(x, y, self.seq_length)  # 重塑为序列格式

        if self.is_train:
            # 删除所有没有测量径流的记录（-999表示缺失值）
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # 删除所有径流为NaN的样本
            if np.sum(np.isnan(y)) > 0:
                print(f"在流域 {self.basin} 中由于NaN值删除了一些记录")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # 在归一化之前存储径流的标准差（用于NSELoss）
            self.q_std = np.std(y)

            # 归一化径流目标值
            y = normalize_features(y, variable='output')

        # 将数组转换为PyTorch张量
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _load_attributes(self) -> torch.Tensor:
        """加载并归一化流域属性"""
        # 从数据库加载流域属性
        df = load_attributes(self.db_path, [self.basin], drop_lat_lon=True)

        # 对属性数据进行归一化
        df = (df - self.attribute_means) / self.attribute_stds

        # 存储属性名称
        self.attribute_names = df.columns

        # 将特征存储为PyTorch张量
        attributes = df.loc[df.index == self.basin].values
        return torch.from_numpy(attributes.astype(np.float32))


class CamelsH5(Dataset):
    """用于处理预打包的HDF5数据库文件的PyTorch数据集类。

    应仅与`papercode.utils`模块中`create_h5_files`处理的文件结合使用。

    参数
    ----------
    h5_file : PosixPath
        包含打包数据的hdf5文件路径
    basins : List
        包含8位USGS测站ID的列表
    db_path : str
        包含流域特征的sqlite3数据库文件路径
    concat_static : bool
        如果为True，在每个时间步将流域特征添加到气象强迫输入数据中，默认为False
    cache : bool, 可选
        如果为True，将所有数据加载到内存中，默认为False
    no_static : bool, 可选
        如果为True，不向输入添加流域属性，默认为False
    """

    def __init__(self,
                 h5_file: PosixPath,
                 basins: List,
                 db_path: str,
                 concat_static: bool = False,
                 cache: bool = False,
                 no_static: bool = False):
        self.h5_file = h5_file
        self.basins = basins
        self.db_path = db_path
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static

        # 流域属性的占位符
        self.df = None
        self.attribute_means = None
        self.attribute_stds = None
        self.attribute_names = None

        # 如果启用缓存，预加载所有数据
        if self.cache:
            (self.x, self.y, self.sample_2_basin, self.q_stds) = self._preload_data()

        # 将属性加载到数据框中
        self._load_attributes()

        # 一次性确定样本数量
        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        """返回数据集中的样本数量"""
        return self.num_samples

    def __getitem__(self, idx: int):
        """根据索引获取单个样本"""
        if self.cache:
            # 如果数据已缓存，直接从内存中获取
            x = self.x[idx]
            y = self.y[idx]
            basin = self.sample_2_basin[idx]  # 样本对应的流域ID
            q_std = self.q_stds[idx]  # 该样本的径流标准差
        else:
            # 否则从HDF5文件中读取
            with h5py.File(self.h5_file, 'r') as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx]
                basin = basin.decode("ascii")  # 将字节字符串解码为普通字符串
                q_std = f["q_stds"][idx]

        if not self.no_static:
            # 从数据框中获取属性并创建2D数组副本
            attributes = self.df.loc[self.df.index == basin].values

            if self.concat_static:
                # 如果使用静态特征连接，将属性重复到序列长度
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                # 将气象观测值与静态属性结合
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)
            else:
                attributes = torch.from_numpy(attributes.astype(np.float32))

        # 转换为PyTorch张量
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        q_std = torch.from_numpy(q_std)

        # 根据配置返回不同的数据组合
        if self.no_static:
            return x, y, q_std
        else:
            if self.concat_static:
                return x, y, q_std
            else:
                return x, attributes, y, q_std

    def _preload_data(self):
        """预加载所有数据到内存中（用于缓存模式）"""
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]  # 解码所有流域ID字符串
            q_stds = f["q_stds"][:]
        return x, y, str_arr, q_stds

    def _get_basins(self):
        """获取数据集中所有唯一的流域ID"""
        if self.cache:
            basins = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, 'r') as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        """加载并归一化流域属性"""
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)

        # 存储均值和标准差
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()

        # 对数据进行归一化
        df = (df - self.attribute_means) / self.attribute_stds

        self.attribute_names = df.columns
        self.df = df

    def get_attribute_means(self) -> pd.Series:
        """返回流域属性的均值
        
        返回
        -------
        pd.Series
            包含每个流域属性的均值
        """
        return self.attribute_means

    def get_attribute_stds(self) -> pd.Series:
        """返回流域属性的标准差
        
        返回
        -------
        pd.Series
            包含每个流域属性的标准差
        """
        return self.attribute_stds