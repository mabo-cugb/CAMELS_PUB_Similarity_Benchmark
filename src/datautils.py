
import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit  # 用于加速Python函数的JIT编译器

# 本研究中忽略的CAMELS流域特征
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

# 在1999年10月01日至2008年09月30日期间所有流域计算的Maurer数据均值和标准差 降水 太阳辐射 最高温度 最低温度 水汽压
SCALER = {
    'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
    'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672])
}


def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """从txt文件加载流域特征并将其存储到sqlite3数据库表中
    
    Parameters
    ----------
    camels_root : PosixPath
        CAMELS数据集主目录的路径
    db_path : str, optional
        数据库文件应保存的路径。如果为None，则将数据库存储在此代码库主文件夹的`data`目录中, 默认为None
    
    Raises
    ------
    RuntimeError
        如果找不到CAMELS属性文件夹
    """
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"在 {attributes_path} 找不到属性文件夹")

    txt_files = attributes_path.glob('camels_*.txt')

    # 将所有属性读入一个大的数据框
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # 将huc列转换为双数字符串
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # 插入数据库
        df.to_sql('basin_attributes', conn)

    print(f"成功将流域属性存储到 {db_path}。")


def load_attributes(db_path: str,
                    basins: List,
                    drop_lat_lon: bool = True,
                    keep_features: List = None) -> pd.DataFrame:
    """从数据库文件加载属性到DataFrame

    Parameters
    ----------
    db_path : str
        sqlite3数据库文件路径
    basins : List
        包含8位USGS测站ID的列表
    drop_lat_lon : bool
        如果为True，从最终数据框中删除纬度和经度列，默认为True
    keep_features : List
        如果传递了列表，将返回包含这些特征的pd.DataFrame。默认情况下，
        返回包含用于训练的特征的pd.DataFrame。

    Returns
    -------
    pd.DataFrame
        属性在pandas DataFrame中。索引是USGS测站ID。纬度和经度被转换为单位球上的x, y, z坐标。
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # 删除数据集中不包含的流域行
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # 删除经纬度列
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # 删除无效属性
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """使用全局预计算的统计量对特征进行归一化

    Parameters
    ----------
    feature : np.ndarray
        要归一化的数据
    variable : str
        ['inputs', 'output']之一，其中`inputs`表示`feature`输入是模型
        输入（气象强迫数据），`output`表示`feature`输入是径流值。

    Returns
    -------
    np.ndarray
        归一化后的特征

    Raises
    ------
    RuntimeError
        如果`variable`既不是'inputs'也不是'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"未知的变量类型 {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """使用全局预计算的统计量对特征进行反归一化

    Parameters
    ----------
    feature : np.ndarray
        要反归一化的数据
    variable : str
        ['inputs', 'output']之一，其中`inputs`表示`feature`输入是模型
        输入（气象强迫数据），`output`表示`feature`输入是径流值。

    Returns
    -------
    np.ndarray
        反归一化后的特征

    Raises
    ------ 
    RuntimeError
        如果`variable`既不是'inputs'也不是'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"未知的变量类型 {variable}")

    return feature


@njit  # 使用numba JIT编译加速计算
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """将数据重塑为LSTM多对一输入样本

    Parameters
    ----------
    x : np.ndarray
        形状为[num_samples, num_features]的输入特征
    y : np.ndarray
        形状为[num_samples, 1]的输出特征
    seq_length : int
        请求的输入序列长度。

    Returns
    -------
    x_new: np.ndarray
        重塑后的输入特征，形状为[num_samples*, seq_length, num_features]，其中 
        num_samples* 等于 num_samples - seq_length + 1，因为开头需要预热启动
    y_new: np.ndarray
        x_new中每个样本的目标值
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
    """从文本文件加载Maurer强迫数据。

    Parameters
    ----------
    camels_root : PosixPath
        CAMELS数据集主目录的路径
    basin : str
        8位USGS测站ID

    Returns
    -------
    df : pd.DataFrame
        包含Maurer强迫数据的DataFrame
    area: int
        流域面积（从强迫文件的标题中读取）

    Raises
    ------
    RuntimeError
        如果找不到强迫文件。
    """
    forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'在 {file_path} 找不到流域 {basin} 的文件')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep='\s+', header=3)
    # 检查列名是 'Month' 还是 'Mnth'
    # 检查列名，处理不同的列名情况
    if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
        dates = (df.Year.map(str) + "/" + df.Month.map(str) + "/" + df.Day.map(str))
    elif 'Year' in df.columns and 'Mnth' in df.columns and 'Day' in df.columns:
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    else:
        # 如果以上都不匹配，打印列名以便调试
        print(f"警告: 数据文件 {file_path} 的列名不标准: {df.columns.tolist()}")
        # 尝试使用位置索引（假设列顺序是固定的）
        if len(df.columns) >= 3:
            dates = (df.iloc[:, 0].map(str) + "/" + df.iloc[:, 1].map(str) + "/" + df.iloc[:, 2].map(str))
        else:
            raise RuntimeError(f"无法解析数据文件 {file_path} 的日期列")
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # 从文件头加载流域面积
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """加载径流数据

    Parameters
    ----------
    camels_root : PosixPath
        CAMELS数据集主目录的路径
    basin : str
        8位USGS测站ID
    area : int
        流域面积，用于将径流归一化为mm/day

    Returns
    -------
    pd.Series
        包含径流值的Series。

    Raises
    ------
    RuntimeError
        如果找不到径流文件。
    """
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'在 {file_path} 找不到流域 {basin} 的文件')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # 将径流从立方英尺/秒归一化为毫米/天
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs
