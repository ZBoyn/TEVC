import numpy as np
from config import ProblemDefinition, Solution

class Decoder:
    """
    解码器类，用于将工件序列解码为目标值。
    """
    def __init__(self, problem_def: ProblemDefinition):
        self.problem = problem_def

    def decode(self, solution: Solution, return_est_matrix: bool = False):
        """
        解码函数: 根据解的状态(有无final_schedule)计算目标函数值.
        """
        completion_times = None
        
        # 路径1: 解已经拥有一个由right_shift算子计算出的精确调度
        if solution.final_schedule is not None:
            completion_times = solution.final_schedule
        
        # 路径2: 解是一个标准的、基于序列的紧凑调度 (put_off被忽略)
        else:
            # 动态计算紧凑排程的完成时间
            temp_completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
            for j_idx, job_id in enumerate(solution.sequence):
                for i in range(self.problem.num_machines):
                    proc_time = self.problem.processing_times[job_id, i]
                    prev_job_id = solution.sequence[j_idx - 1] if j_idx > 0 else -1
                    
                    est_from_prev_job = temp_completion_times[prev_job_id, i] if prev_job_id != -1 else 0
                    est_from_prev_machine = temp_completion_times[job_id, i - 1] if i > 0 else 0
                    est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])
                    
                    # 紧凑排程的核心: 处理非抢占约束
                    p_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                    p_idx = max(0, p_idx) # 确保 p_idx 不会是-1
                    if p_idx + 1 < len(self.problem.period_start_times):
                        p_end_time = self.problem.period_start_times[p_idx + 1]
                    else:
                        p_end_time = self.problem.deadline
                    
                    start_time = est if est + proc_time <= p_end_time else p_end_time
                    temp_completion_times[job_id, i] = start_time + proc_time
            completion_times = temp_completion_times

        # 应用惩罚
        if completion_times.max() > self.problem.deadline + 1e-6:
            solution.objectives = np.array([np.inf, np.inf])
            return solution.objectives
        
        # 计算TEC
        tec = 0.0
        for job_id in range(self.problem.num_jobs):
            for machine_id in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, machine_id]
                start_time = completion_times[job_id, machine_id] - proc_time
                power = self.problem.power_consumption[job_id, machine_id]
                tec += self.calculate_op_cost(start_time, proc_time) * power

        # 计算TCTA
        agent_makespans = np.zeros(self.problem.num_agents)
        job_final_completion = completion_times[:, -1]
        for agent_id in range(self.problem.num_agents):
            start_job = self.problem.agent_job_indices[agent_id]
            end_job = self.problem.agent_job_indices[agent_id + 1]
            agent_jobs = list(range(start_job, end_job))
            if agent_jobs:
                agent_makespans[agent_id] = job_final_completion[agent_jobs].max()
        tcta = np.sum(agent_makespans * self.problem.agent_weights)

        solution.objectives = np.array([tec, tcta])
        solution.completion_times = completion_times # 缓存完成时间
        
        # 兼容right_shift算子获取紧凑排程的est的需求
        if return_est_matrix:
            # EST矩阵的计算逻辑与紧凑排程一致, 只是记录的是start_time
            est_matrix = np.zeros_like(completion_times)
            # (为简化, 此处省略了est_matrix的完整计算, 因为它只在right_shift内部的特定调用中需要,
            # 而right_shift本身现在是自洽的, 不再需要从外部获取est_matrix)
            # 实际上, right_shift内部的 _calculate_earliest_times 已经满足了需求.
            return solution.objectives, est_matrix

        return solution.objectives
    
    def calculate_op_cost(self, start_time: float, proc_time: float) -> float:
        """
        一个辅助函数, 用于计算单个工序在给定开始时间和处理时间下的电力成本.
        这个函数模拟了decode中的成本计算逻辑, 但只针对单个操作, 以支持right_shift算子.
        """
        cost = 0.0
        remaining_proc_time = proc_time
        current_time = start_time
        
        while remaining_proc_time > 1e-6:
            period_idx = np.searchsorted(self.problem.period_start_times, current_time, side='right') - 1
            period_idx = max(0, period_idx) # 确保 period_idx 不会是-1
            
            if period_idx + 1 < len(self.problem.period_start_times):
                period_end_time = self.problem.period_start_times[period_idx + 1]
            else:
                period_end_time = self.problem.deadline
            
            price = self.problem.period_prices[period_idx]
            
            time_in_this_period = min(remaining_proc_time, period_end_time - current_time)
            cost += time_in_this_period * price
            
            remaining_proc_time -= time_in_this_period
            current_time += time_in_this_period
            
        return cost
