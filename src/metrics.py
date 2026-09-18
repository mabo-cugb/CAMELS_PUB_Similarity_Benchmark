import numpy as np


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe-Effiency
    纳什-萨克利夫效率系数，用于评估水文模型模拟效果
    
    Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组
    
    Returns
    -------
    float
        纳什-萨克利夫效率系数
    
    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    RuntimeError
        如果观测值中所有值都相等
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten() # 展平为一维数组
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。") # 抛出一个运行时错误，并显示错误信息

    # 分数项的分母
    denominator = np.sum((obs - np.mean(obs))**2)

    # 这将导致除以零错误，nse定义为负无穷
    if denominator == 0:
        msg = [
            "当观测值中所有值都相等时，纳什-萨克利夫效率系数未定义。",
            " 也许您应该使用均方误差代替。"
        ]
        raise RuntimeError("".join(msg))

    # 分数项的分子
    numerator = np.sum((sim - obs)**2)

    # 计算NSE
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_alpha_nse(obs: np.ndarray, sim: np.ndarray) -> float: # 表示模拟值与观测值变异性的比率
    """NSE的Alpha分解，参见Gupta等人2009年的研究

    Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组

    Returns
    -------
    float
        NSE的Alpha分解

    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。")

    return np.std(sim) / np.std(obs)


def calc_beta_nse(obs: np.ndarray, sim: np.ndarray) -> float: # 表示标准化后的均值偏差
    """NSE的Beta分解。参见Gupta等人2009年的研究
    
    Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组

    Returns
    -------
    float
        NSE的Beta分解

    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。")

    return (np.mean(sim) - np.mean(obs)) / np.std(obs)


def calc_fdc_fms(obs: np.ndarray, sim: np.ndarray, m1: float = 0.2, m2: float = 0.7) -> float:
    """计算流量历时曲线中段斜率偏差
    
    Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组
    m1 : float, optional
        中段下限。必须在(0,1)范围内, 默认为0.2
    m2 : float, optional
        中段上限。必须在(0,1)范围内, 默认为0.7
    
    Returns
    -------
    float
        流量历时曲线中段斜率的偏差（Yilmaz 2018）。
    
    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    RuntimeError
        如果`m1`不在(0,1)范围内
    RuntimeError
        如果`m2`不在(0,1)范围内
    RuntimeError
        如果`m1` >= `m2`
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。")

    if (m1 <= 0) or (m1 >= 1):
        raise RuntimeError("m1必须在(0,1)范围内")

    if (m2 <= 0) or (m2 >= 1):
        raise RuntimeError("m2必须在(0,1)范围内")

    if m1 >= m2:
        raise RuntimeError("m1必须小于m2")

    # 出于数值计算原因，将0替换为1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # 按降序排序
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # 分段计算fms
    qsm1 = np.log(sim[np.round(m1 * len(sim)).astype(int)] + 1e-6)
    qsm2 = np.log(sim[np.round(m2 * len(sim)).astype(int)] + 1e-6)
    qom1 = np.log(obs[np.round(m1 * len(obs)).astype(int)] + 1e-6)
    qom2 = np.log(obs[np.round(m2 * len(obs)).astype(int)] + 1e-6)

    fms = ((qsm1 - qsm2) - (qom1 - qom2)) / (qom1 - qom2 + 1e-6)

    return fms * 100


def calc_fdc_fhv(obs: np.ndarray, sim: np.ndarray, h: float = 0.02) -> float:
    """流量历时曲线的峰值流量偏差（Yilmaz 2018）。
    
    Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组
    h : float, optional
        被视为峰值流量的流量比例。必须在(0,1)范围内, 默认为0.02
    
    Returns
    -------
    float
        峰值流量的偏差
    
    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    RuntimeError
        如果`h`不在(0,1)范围内
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h必须在(0,1)范围内")

    # 按降序排序
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # 仅选取前h个流量值
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

    return fhv * 100


def calc_fdc_flv(obs: np.ndarray, sim: np.ndarray, l: float = 0.7) -> float:
    
    
    """Parameters
    ----------
    obs : np.ndarray
        包含径流观测值的数组
    sim : np.ndarray
        包含径流模拟值的数组
    l : float, optional
        流量历时曲线的上限。例如0.7表示考虑底部30%的流量作为低流量, 默认为0.7
    
    Returns
    -------
    float
        低流量的偏差。
    
    Raises
    ------
    RuntimeError
        如果`obs`和`sim`长度不一致
    RuntimeError
        如果`l`不在(0,1)范围内
    """
    # 确保在同一维度上计算指标
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs和sim必须具有相同长度。")

    if (l <= 0) or (l >= 1):
        raise RuntimeError("l必须在(0,1)范围内")

    # 出于数值计算原因，将0替换为1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # 按降序排序
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # 仅选取底部(1-l)部分的流量值
    obs = obs[np.round(l * len(obs)).astype(int):]
    sim = sim[np.round(l * len(sim)).astype(int):]

    # 将值转换为对数尺度
    obs = np.log(obs + 1e-6)
    sim = np.log(sim + 1e-6)

    # 分段计算flv
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100
