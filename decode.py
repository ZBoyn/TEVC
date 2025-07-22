import numpy as np
from config import ProblemDefinition, Solution

class Decoder:
    """
    解码器类，用于将工件序列解码为目标值。
    """
    def __init__(self, problem_def: ProblemDefinition):
        self.problem = problem_def

    def decode(self, solution: Solution) -> np.ndarray:
        """
        对给定的解进行解码和评估。

        Args:
            solution (Solution): 包含工件序列和put_off矩阵的解。

        Returns:
            np.ndarray: 一个包含两个目标值的数组 [TEC, TCTA]。
        """
        solution.put_off = np.round(solution.put_off).astype(int)
        
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        operation_periods = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)

        # 计算每个工件在每台机器上的完成时间
        for j_idx, job_id in enumerate(solution.sequence):
            for machine_id in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, machine_id]

                # 1. 计算无任何推迟时的理论最早开始时间 (EST)
                # 来自前序工件的约束 (在当前机器上)
                prev_job_id = solution.sequence[j_idx - 1] if j_idx > 0 else -1
                est_from_prev_job = completion_times[prev_job_id, machine_id] if prev_job_id != -1 else 0
                # 来自本工件的前道工序的约束 (在前一台机器上)
                est_from_prev_machine = completion_times[job_id, machine_id - 1] if machine_id > 0 else 0
                # 结合工件自身释放时间
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])

                # 2. 应用时段推迟put_off
                # 找到EST本应该在哪个时段开始
                base_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                
                # 根据put_off矩阵计算实际开始时段
                target_period_idx = min(base_period_idx + solution.put_off[job_id, machine_id], self.problem.num_periods - 1)
                
                # 目标时段的开始时间
                target_period_start_time = self.problem.period_start_times[target_period_idx]
                
                # 得到延迟后的EST
                delayed_est = max(target_period_start_time, est)
                
                # 3. 应用电价时段内的紧凑排列约束
                current_period_idx = np.searchsorted(self.problem.period_start_times, delayed_est, side='right') - 1
                if current_period_idx + 1 < len(self.problem.period_start_times):
                    period_end_time = self.problem.period_start_times[current_period_idx + 1]
                else:
                    # 修复: 最后一个时段的结束时间是全局deadline
                    period_end_time = self.problem.deadline
                
                if delayed_est + proc_time <= period_end_time:
                    actual_start_time = delayed_est
                else:
                    # 如果当前时段无法容纳该工件, 则推迟到下一个可用时段
                    # 修复: 必须检查是否存在下一个时段
                    if current_period_idx + 1 < len(self.problem.period_start_times):
                        actual_start_time = self.problem.period_start_times[current_period_idx + 1]
                    else:
                        # 如果已是最后一个时段且无法容纳, 则接受溢出,
                        # 后续的deadline检查会将其标记为不可行解.
                        actual_start_time = delayed_est
                
                completion_times[job_id, machine_id] = actual_start_time + proc_time
                final_period_idx = np.searchsorted(self.problem.period_start_times, actual_start_time, side='right') - 1
                operation_periods[job_id, machine_id] = final_period_idx
        
        # 应用惩罚机制
        if completion_times.max() > self.problem.deadline + 1e-6:
            solution.objectives = np.array([np.inf, np.inf])
            return solution.objectives

        tec = 0.0
        prices = self.problem.period_prices[operation_periods]
        energy_matrix = self.problem.power_consumption * self.problem.processing_times
        tec = np.sum(prices * energy_matrix)
                
        agent_makespans = np.zeros(self.problem.num_agents)
        job_final_completion = completion_times[:, -1]  # 最后一台机器的完成时间
        for agent_id in range(self.problem.num_agents):
            start_job = self.problem.agent_job_indices[agent_id]
            end_job = self.problem.agent_job_indices[agent_id + 1]
            agent_jobs = list(range(start_job, end_job))
            if agent_jobs:
                agent_makespans[agent_id] = job_final_completion[agent_jobs].max()
        tcta = np.sum(agent_makespans * self.problem.agent_weights)
        
        solution.objectives = np.array([tec, tcta])
        return solution.objectives
