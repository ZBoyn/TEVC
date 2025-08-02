from typing import Tuple, Set
import numpy as np
from dataclasses import dataclass, field
from pro_def import ProblemDefinition
from bb_node import SearchNode

class BoundCalculator:
    """封装下界计算逻辑"""
    @staticmethod
    def _get_actual_schedule_completion_times(
        sequence: Tuple[int, ...], 
        put_off_decisions: Tuple[int, ...], 
        problem: ProblemDefinition
    ) -> np.ndarray:
        """获取实际调度完成时间

        Args:
            sequence (Tuple[int, ...]): 调度序列
            put_off_decisions (Tuple[int, ...]): 推迟决策
            problem (ProblemDefinition): 问题定义

        Returns:
            np.ndarray: 实际调度完成时间
        """
        if not sequence:
            return np.zeros((problem.num_jobs, problem.num_machines))
        
        completion_times = np.zeros((problem.num_jobs, problem.num_machines))
        seq_to_put_off_map = {job_id: put_off for job_id, put_off in zip(sequence, put_off_decisions)}
        # 遍历调度序列 计算每个工件在每台机器上的完成时间
        for j_idx, job_id in enumerate(sequence):
            for i in range(problem.num_machines):
                proc_time = problem.processing_times[job_id, i]
                est_from_prev_job = completion_times[sequence[j_idx - 1], i] if j_idx > 0 else 0
                est_from_prev_machine = completion_times[job_id, i - 1] if i > 0 else 0
                start_time = max(est_from_prev_job, est_from_prev_machine, problem.release_times[job_id])
                if i == 0 and seq_to_put_off_map.get(job_id, 0) == 1:
                    current_period_idx = np.searchsorted(problem.period_start_times, start_time, side='right') - 1
                    if current_period_idx + 1 < problem.num_periods:
                        start_time = max(start_time, problem.period_start_times[current_period_idx + 1])
                actual_start_time = start_time
                while True:
                    p_idx = np.searchsorted(problem.period_start_times, actual_start_time, side='right') - 1
                    p_idx = max(0, p_idx)
                    p_end_time = problem.period_start_times[p_idx + 1] if p_idx + 1 < len(problem.period_start_times) else float('inf')
                    if actual_start_time + proc_time <= p_end_time: break
                    else:
                        if p_idx + 1 < len(problem.period_start_times): actual_start_time = problem.period_start_times[p_idx + 1]
                        else: break
                completion_times[job_id, i] = actual_start_time + proc_time
        return completion_times

    @staticmethod
    def _calculate_op_cost(start_time: float, proc_time: float, power: float, problem: ProblemDefinition) -> float:
        cost, remaining_proc_time, current_time = 0.0, proc_time, start_time
        if current_time < problem.period_start_times[0]:
            time_to_process = min(remaining_proc_time, problem.period_start_times[0] - current_time)
            cost += time_to_process * problem.period_prices[0] * power
            remaining_proc_time -= time_to_process
            current_time += time_to_process
        while remaining_proc_time > 1e-6:
            period_idx = np.searchsorted(problem.period_start_times, current_time, side='right') - 1
            if period_idx >= problem.num_periods - 1:
                cost += remaining_proc_time * problem.period_prices[problem.num_periods - 1] * power
                break
            period_end_time = problem.period_start_times[period_idx + 1]
            time_to_process = min(remaining_proc_time, period_end_time - current_time)
            cost += time_to_process * problem.period_prices[period_idx] * power
            remaining_proc_time -= time_to_process
            current_time += time_to_process
        return cost

    @classmethod
    def calculate_tec_lower_bound(cls, node: SearchNode, problem: ProblemDefinition) -> float:
        actual_completion_times = cls._get_actual_schedule_completion_times(node.sequence, node.put_off_decisions, problem)
        scheduled_cost = 0.0
        for job_id in node.sequence:
            for machine_id in range(problem.num_machines):
                proc_time, power = problem.processing_times[job_id, machine_id], problem.power_consumption[job_id, machine_id]
                start_time = actual_completion_times[job_id, machine_id] - proc_time
                scheduled_cost += cls._calculate_op_cost(start_time, proc_time, power, problem)
        if not node.unscheduled_jobs:
            return scheduled_cost
        unscheduled_cost_lb = 0.0
        machine_ready_times = actual_completion_times.max(axis=0) if node.sequence else np.zeros(problem.num_machines)
        unscheduled_jobs_list = list(node.unscheduled_jobs)
        for i in range(problem.num_machines):
            job_energies = [{'id': jid, 'energy': problem.power_consumption[jid, i], 'proc_time': problem.processing_times[jid, i]} for jid in unscheduled_jobs_list]
            job_energies.sort(key=lambda x: x['energy'], reverse=True)
            temp_machine_time = machine_ready_times[i]
            for job_info in job_energies:
                unscheduled_cost_lb += cls._calculate_op_cost(temp_machine_time, job_info['proc_time'], job_info['energy'], problem)
                temp_machine_time += job_info['proc_time']
        return scheduled_cost + unscheduled_cost_lb

    @classmethod
    def calculate_tcta_lower_bound(cls, node: SearchNode, problem: ProblemDefinition) -> float:
        actual_completion_times = cls._get_actual_schedule_completion_times(node.sequence, node.put_off_decisions, problem)
        machine_ready_times = actual_completion_times.max(axis=0) if node.sequence else np.zeros(problem.num_machines)
        machine_lbs = np.zeros(problem.num_machines)
        
        # PADA rule simulation (simplified: using SPT as a proxy for agent priority)
        # A full PADA implementation is complex, SPT is a common heuristic.
        unscheduled_jobs_list = list(node.unscheduled_jobs)
        unscheduled_jobs_list.sort(key=lambda jid: problem.job_total_processing_times[jid])

        # Calculate LB for each machine
        for m in range(problem.num_machines):
            temp_machine_time = machine_ready_times[m]
            # Calculate completion times for unscheduled jobs on this machine
            for job_id in unscheduled_jobs_list:
                est = max(temp_machine_time, problem.release_times[job_id])
                # A very loose LB for previous machine completion time
                est_from_prev_machines = sum(problem.processing_times[job_id, :m])
                est = max(est, est_from_prev_machines)
                temp_machine_time = est + problem.processing_times[job_id, m]
            
            # Tail correction for all but the last machine
            if m < problem.num_machines - 1 and unscheduled_jobs_list:
                min_remaining_proc_time = min(problem.processing_times[jid, m+1:].sum() for jid in unscheduled_jobs_list)
                machine_lbs[m] = temp_machine_time + min_remaining_proc_time
            else:
                machine_lbs[m] = temp_machine_time
        
        # Total LB is max over machines, plus scheduled part
        scheduled_makespans = actual_completion_times[:, -1] if node.sequence else np.array([0])
        overall_lb = max(np.max(machine_lbs), np.max(scheduled_makespans))
        
        # For TCTA, we should return a value proportional to makespan
        return overall_lb * np.sum(problem.agent_weights)

class PruningRules:
    """封装剪枝规则"""
    @staticmethod
    def should_prune(node: SearchNode, next_job: int, is_put_off: bool, problem: ProblemDefinition, objective_type: str) -> bool:
        """
        检查在当前节点(node)的基础上，调度下一个工件(next_job)
        并采用特定推迟决策(is_put_off)是否应被剪枝。
        """
        if objective_type == 'TEC' and is_put_off:
            last_completion_time_m1 = node.completion_times_m1[-1] if node.completion_times_m1.size > 0 else 0
            natural_start_time_m1 = max(last_completion_time_m1, problem.release_times[next_job])
            natural_period_idx = np.searchsorted(problem.period_start_times, natural_start_time_m1, side='right') - 1
            natural_period_idx = max(0, natural_period_idx)
            if natural_period_idx == problem.cheapest_period_index:
                cheapest_period_end_time = (
                    problem.period_start_times[natural_period_idx + 1] 
                    if natural_period_idx + 1 < problem.num_periods 
                    else float('inf')
                )
                proc_time_m1 = problem.processing_times[next_job, 0]
                if natural_start_time_m1 + proc_time_m1 <= cheapest_period_end_time:
                    return True
        return False