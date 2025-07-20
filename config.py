import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ProblemDefinition:
    """问题定义 封装所有问题的固定参数"""
    # 核心参数
    processing_times: np.ndarray    # P[N, M]: 工件 x 机器
    power_consumption: np.ndarray   # E[N, M]: 工件 x 机器
    release_times: np.ndarray       # R[N]: 工件释放时间
    agent_job_indices: np.ndarray   # AN[A+1]: 每个代理的起始工件索引
    agent_weights: np.ndarray       # AW[A]: 每个代理的权重
    period_start_times: np.ndarray  # U[K+1]: 电价时段的开始时间
    period_prices: np.ndarray       # W[K]: 每个时段的电价

    # 派生参数
    num_jobs: int = field(init=False)
    num_machines: int = field(init=False)
    num_agents: int = field(init=False)
    num_periods: int = field(init=False)
    
    # 映射表和属性
    job_to_agent_map: Dict[int, int] = field(default_factory=dict, init=False)
    agent_priority: List[int] = field(default_factory=list, init=False) # 按优劣排序的代理ID
    job_total_energy: np.ndarray = field(init=False) # 各工件总能耗，用于H2
    cheapest_period_index: int = field(init=False) # 最低电价时段的索引
    
    def __post_init__(self):
        """初始化派生参数"""
        self.num_jobs, self.num_machines = self.processing_times.shape
        self.num_agents = len(self.agent_weights)
        self.num_periods = len(self.period_prices)
        
        # 创建工件到代理的映射
        for agent_idx in range(self.num_agents):
            start_job_idx = self.agent_job_indices[agent_idx]
            end_job_idx = self.agent_job_indices[agent_idx + 1]
            for job_idx in range(start_job_idx, end_job_idx):
                self.job_to_agent_map[job_idx] = agent_idx
        
        self.job_total_energy = (self.processing_times * self.power_consumption).sum(axis=1) # 各工件总能耗
        self.cheapest_period_index = np.argmin(self.period_prices)
    
@dataclass
class Solution:
    """定义一个解(individual)的结构"""
    sequence: np.ndarray
    put_off: np.ndarray
    
    objectives: np.ndarray = field(default_factory=lambda: np.full(2, np.inf)) # [TEC, TCTA]
    rank: int = -1
    crowding_distance: float = 0.0