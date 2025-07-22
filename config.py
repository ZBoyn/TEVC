import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

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
    deadline: float = field(init=False)
    
    # 映射表和属性
    job_to_agent_map: Dict[int, int] = field(default_factory=dict, init=False)
    agent_priority: List[int] = field(default_factory=list, init=False) # 按优劣排序的代理ID
    job_total_energy: np.ndarray = field(init=False) # 各工件总能耗，用于H2
    job_total_processing_times: np.ndarray = field(init=False) # 各工件总加工时间, 用于NEH
    job_spt_order: np.ndarray = field(init=False) # 按总加工时间升序排列的工件索引
    cheapest_period_index: int = field(init=False) # 最低电价时段的索引
    
    def __post_init__(self):
        """初始化派生参数"""
        self.num_jobs, self.num_machines = self.processing_times.shape
        self.num_agents = len(self.agent_weights)
        self.num_periods = len(self.period_prices)
        
        # 问题的最终截止日期是最后一个时段的结束时间
        if len(self.period_start_times) > self.num_periods:
            self.deadline = self.period_start_times[-1]
        else:
            # 如果数据格式不规范, 做一个兼容处理
            # 假设最后一个时段的持续时间与其他时段平均持续时间相同
            if self.num_periods > 1:
                avg_duration = (self.period_start_times[-1] - self.period_start_times[0]) / (self.num_periods - 1)
                self.deadline = self.period_start_times[-1] + avg_duration
            else:
                self.deadline = self.period_start_times[0] * 2 # 兜底
        
        # 创建工件到代理的映射
        for agent_idx in range(self.num_agents):
            start_job_idx = self.agent_job_indices[agent_idx]
            end_job_idx = self.agent_job_indices[agent_idx + 1]
            for job_idx in range(start_job_idx, end_job_idx):
                self.job_to_agent_map[job_idx] = agent_idx
        
        self.job_total_energy = (self.processing_times * self.power_consumption).sum(axis=1) # 各工件总能耗
        self.job_total_processing_times = self.processing_times.sum(axis=1) # 各工件总加工时间
        self.job_spt_order = np.argsort(self.job_total_processing_times) # SPT: Shortest Processing Time
        self.cheapest_period_index = np.argmin(self.period_prices)
    
@dataclass
class Solution:
    """定义一个解(individual)的结构"""
    sequence: np.ndarray
    put_off: np.ndarray
    
    final_schedule: np.ndarray = None # 由right_shift算子生成
    completion_times: np.ndarray = None # 用于缓存解码后的完成时间
    
    objectives: np.ndarray = field(default_factory=lambda: np.full(2, np.inf)) # [TEC, TCTA]
    rank: int = -1
    crowding_distance: float = 0.0
    
    def copy(self):
        """创建解的一个深拷贝"""
        return Solution(sequence=self.sequence.copy(),
                        put_off=self.put_off.copy(),
                        final_schedule=self.final_schedule.copy() if self.final_schedule is not None else None,
                        completion_times=self.completion_times.copy() if self.completion_times is not None else None,
                        objectives=self.objectives.copy(),
                        rank=self.rank,
                        crowding_distance=self.crowding_distance)