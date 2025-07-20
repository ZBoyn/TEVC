import numpy as np
import math

def generate_weight_vectors(k: int) -> np.ndarray:
    """生成 k 组归一化的随机权重向量, 确保 w1 + w2 = 1"""
    r = np.random.rand(k, 2)
    r[r.sum(axis=1) == 0] = 1e-9
    row_sums = r.sum(axis=1, keepdims=True)
    weights = r / row_sums
    return weights

def calculate_objectives(sequence: np.ndarray, processing_times: np.ndarray, electricity_prices: np.ndarray) -> tuple:
    """
    计算一个工件序列的原始目标值: 总电费(TEC)和总完工时间(TCTA)
    (此函数保持不变, 您仍需实现其内部逻辑)
    """
    # --- 请在此处实现您的调度模拟逻辑 ---
    # 为了演示，我们继续使用随机值
    num_jobs, num_machines = processing_times.shape
    simulated_tec = np.random.uniform(500, 2000)
    simulated_tcta = np.random.uniform(np.sum(processing_times) / num_machines, np.sum(processing_times))
    return simulated_tec, simulated_tcta

def calc_fitness(TEC: float, TCTA: float, weight_vectors: np.ndarray) -> float:
    """
    计算最终的加权适应度
    """
    if TEC <= 0: TEC = 1e-6
    if TCTA <= 0: TCTA = 1e-6
    scaled_tec = TEC / 10**(int(np.log10(TEC)) + 1)
    scaled_tcta = TCTA / 10**(int(np.log10(TCTA)) + 1)
    targets = np.array([scaled_tec, scaled_tcta])
    weighted_sums = (weight_vectors * targets).sum(axis=1)
    return weighted_sums.mean()


def fitness_wrapper(sequence: np.ndarray, **problem_params) -> float:
    """适应度函数封装器

    Args:
        sequence (np.ndarray): 工件序列

    Returns:
        float: 适应度值
        float: 总电费(TEC)
        float: 总完工时间(TCTA)
    """
    processing_times = problem_params['processing_times']
    electricity_prices = problem_params['electricity_prices']
    weight_vectors = problem_params['weight_vectors']
    TEC, TCTA = calculate_objectives(sequence, processing_times, electricity_prices)
    final_fitness = calc_fitness(TEC, TCTA, weight_vectors)
    return final_fitness, TEC, TCTA