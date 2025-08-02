import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Deque
import ast
import os
from collections import deque
from data_loader import load_problem_from_file
# --------------------------------------------------------------------------
# 1. 问题定义
# --------------------------------------------------------------------------
@dataclass
class ProblemDefinition:
    """问题定义 封装所有问题的固定参数"""
    # 核心参数
    processing_times: np.ndarray   # P[N, M]: 工件 x 机器
    power_consumption: np.ndarray  # E[N, M]: 工件 x 机器
    release_times: np.ndarray      # R[N]: 工件释放时间
    agent_job_indices: np.ndarray  # AN[A+1]: 每个代理的起始工件索引
    agent_weights: np.ndarray      # AW[A]: 每个代理的权重
    period_start_times: np.ndarray # U[K+1]: 电价时段的开始时间
    period_prices: np.ndarray      # W[K]: 每个时段的电价

    # 派生参数
    num_jobs: int = field(init=False)
    num_machines: int = field(init=False)
    num_agents: int = field(init=False)
    num_periods: int = field(init=False)
    deadline: float = field(init=False)
    
    # 映射表和属性
    job_to_agent_map: Dict[int, int] = field(default_factory=dict, init=False)
    job_total_energy: np.ndarray = field(init=False) 
    job_total_processing_times: np.ndarray = field(init=False)
    cheapest_period_index: int = field(init=False)
    
    def __post_init__(self):
        """初始化派生参数"""
        self.num_jobs, self.num_machines = self.processing_times.shape
        self.num_agents = len(self.agent_weights)
        self.num_periods = len(self.period_prices)
        
        if len(self.period_start_times) > self.num_periods:
            self.deadline = self.period_start_times[-1]
        else:
            if self.num_periods > 1:
                avg_duration = (self.period_start_times[-1] - self.period_start_times[0]) / (self.num_periods - 1)
                self.deadline = self.period_start_times[-1] + avg_duration
            else:
                self.deadline = self.period_start_times[0] * 2

        for agent_idx in range(self.num_agents):
            start_job_idx = self.agent_job_indices[agent_idx]
            end_job_idx = self.agent_job_indices[agent_idx + 1]
            for job_idx in range(start_job_idx, end_job_idx):
                self.job_to_agent_map[job_idx] = agent_idx
        
        self.job_total_energy = (self.processing_times * self.power_consumption).sum(axis=1)
        self.job_total_processing_times = self.processing_times.sum(axis=1)
        self.cheapest_period_index = np.argmin(self.period_prices)
        
        self.agent_weights = np.ones(self.num_agents)

# --------------------------------------------------------------------------
# 2. 搜索节点定义
# --------------------------------------------------------------------------
@dataclass(order=False)
class SearchNode:
    """定义搜索树中的一个节点，代表一个部分调度方案"""
    sequence: Tuple[int, ...] = field(default_factory=tuple)
    put_off_decisions: Tuple[int, ...] = field(default_factory=tuple)
    unscheduled_jobs: Set[int] = field(default_factory=set)
    completion_times_m1: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_bound_tec: float = 0.0
    lower_bound_tcta: float = 0.0
    priority: float = 0.0
    
    def __lt__(self, other: 'SearchNode') -> bool:
        return self.priority < other.priority

# --------------------------------------------------------------------------
# 3. 下界计算器 (已重构)
# --------------------------------------------------------------------------
class BoundCalculator:
    """封装下界计算逻辑"""
    @staticmethod
    def _get_actual_schedule_completion_times(
        sequence: Tuple[int, ...], 
        put_off_decisions: Tuple[int, ...], 
        problem: ProblemDefinition
    ) -> np.ndarray:
        if not sequence:
            return np.zeros((problem.num_jobs, problem.num_machines))
        completion_times = np.zeros((problem.num_jobs, problem.num_machines))
        seq_to_put_off_map = {job_id: put_off for job_id, put_off in zip(sequence, put_off_decisions)}
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

# --------------------------------------------------------------------------
# 4. 剪枝规则检查器
# --------------------------------------------------------------------------
class PruningRules:
    @staticmethod
    def should_prune(node: SearchNode, next_job: int, is_put_off: bool, problem: ProblemDefinition, objective_type: str) -> bool:
        if objective_type == 'TEC' and is_put_off:
            last_completion_time_m1 = node.completion_times_m1[-1] if node.completion_times_m1.size > 0 else 0
            natural_start_time_m1 = max(last_completion_time_m1, problem.release_times[next_job])
            natural_period_idx = np.searchsorted(problem.period_start_times, natural_start_time_m1, side='right') - 1
            natural_period_idx = max(0, natural_period_idx)
            if natural_period_idx == problem.cheapest_period_index:
                cheapest_period_end_time = problem.period_start_times[natural_period_idx + 1] if natural_period_idx + 1 < problem.num_periods else float('inf')
                if natural_start_time_m1 + problem.processing_times[next_job, 0] <= cheapest_period_end_time:
                    return True
        return False

# --------------------------------------------------------------------------
# 5. 启发式B&B求解器 (已重构)
# --------------------------------------------------------------------------
class HeuristicBBSolver:
    def __init__(self, problem: ProblemDefinition):
        self.problem = problem
        self.bound_calculator = BoundCalculator()
        self.pruning_rules = PruningRules()

    def find_heuristic_solution(self, objective_type: str, verbose: bool = True) -> Optional[Tuple[List[int], np.ndarray]]:
        # 使用双端队列作为回溯栈
        backtrack_stack: Deque[Tuple[SearchNode, List[SearchNode]]] = deque()
        current_node = SearchNode(unscheduled_jobs=set(range(self.problem.num_jobs)))
        
        iteration_count = 0
        while True:
            iteration_count += 1
            if verbose:
                print(f"\n--- Iteration {iteration_count}, Stack Size: {len(backtrack_stack)} ---")
                print(f"Current node: Seq={current_node.sequence}")

            if not current_node.unscheduled_jobs:
                print(f"\n{'='*20} SUCCESS {'='*20}")
                print(f"成功找到一个完整序列，目标: {objective_type}")
                print(f"总迭代次数: {iteration_count}")
                return self._reconstruct_solution(current_node)

            # 1. 生成所有有效的子节点
            children = []
            for job_to_schedule in sorted(list(current_node.unscheduled_jobs)):
                for is_put_off in [False, True]:
                    if self.pruning_rules.should_prune(current_node, job_to_schedule, is_put_off, self.problem, objective_type):
                        continue
                    child_node = self._create_child_node(current_node, job_to_schedule, is_put_off)
                    if child_node:
                        self._calculate_and_set_bounds(child_node, objective_type)
                        children.append(child_node)
            
            # 2. 选择最佳子节点深入，其他子节点存入回溯栈
            if children:
                children.sort() # 按priority升序排序
                best_child = children.pop(0) # 取出最好的
                
                if verbose:
                    print(f"  -> Diving into best child: Seq={best_child.sequence}, Prio={best_child.priority:.2f}")
                
                # 将当前节点和剩余的子节点存入回溯栈
                if children:
                    backtrack_stack.append((current_node, children))
                    if verbose:
                        print(f"  -> Pushed {len(children)} other children to backtrack stack.")

                current_node = best_child
            
            # 3. 如果没有有效子节点，进行回溯
            else:
                if verbose:
                    print(f"  -> Dead end. Backtracking...")
                if not backtrack_stack:
                    print(f"\n{'='*20} FAILURE {'='*20}")
                    print(f"未能找到完整序列，目标: {objective_type}. Backtrack stack is empty.")
                    return None
                
                # 从栈中恢复上一个节点和它的待选子节点列表
                parent_node, remaining_children = backtrack_stack.pop()
                
                if verbose:
                    print(f"  -> Backtracked to node Seq={parent_node.sequence}. Trying next best child.")
                
                # 从剩余的子节点中选择下一个最好的
                best_child = remaining_children.pop(0)
                
                # 如果这个分支还有剩余的子节点，重新入栈
                if remaining_children:
                    backtrack_stack.append((parent_node, remaining_children))

                current_node = best_child

    def _create_child_node(self, parent: SearchNode, job: int, is_put_off: bool) -> Optional[SearchNode]:
        last_completion_time = parent.completion_times_m1[-1] if parent.completion_times_m1.size > 0 else 0
        start_time_m1 = max(last_completion_time, self.problem.release_times[job])
        if is_put_off:
            current_period_idx = np.searchsorted(self.problem.period_start_times, start_time_m1, side='right') - 1
            if current_period_idx + 1 < self.problem.num_periods:
                start_time_m1 = max(start_time_m1, self.problem.period_start_times[current_period_idx + 1])
        completion_time_m1 = start_time_m1 + self.problem.processing_times[job, 0]
        if completion_time_m1 > self.problem.deadline: return None
        new_sequence = parent.sequence + (job,)
        new_put_off = parent.put_off_decisions + (int(is_put_off),)
        new_unscheduled = parent.unscheduled_jobs - {job}
        new_completion_times = np.append(parent.completion_times_m1, completion_time_m1)
        return SearchNode(sequence=new_sequence, put_off_decisions=new_put_off, unscheduled_jobs=new_unscheduled, completion_times_m1=new_completion_times)

    def _calculate_and_set_bounds(self, node: SearchNode, objective_type: str):
        node.lower_bound_tec = self.bound_calculator.calculate_tec_lower_bound(node, self.problem)
        node.lower_bound_tcta = self.bound_calculator.calculate_tcta_lower_bound(node, self.problem)
        node.priority = node.lower_bound_tec if objective_type == 'TEC' else node.lower_bound_tcta

    def _reconstruct_solution(self, leaf_node: SearchNode) -> Tuple[List[int], np.ndarray]:
        sequence = list(leaf_node.sequence)
        put_off_matrix = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        put_off_m1 = np.array(leaf_node.put_off_decisions)
        original_indices_order = np.array(sequence)
        put_off_matrix[original_indices_order, 0] = put_off_m1
        return sequence, put_off_matrix

# --------------------------------------------------------------------------
# 6. 数据加载与主程序入口
# --------------------------------------------------------------------------
def load_problem_g_format(file_path: str) -> ProblemDefinition:
    raw_data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line: continue
                key, value_part = line.split('=', 1)
                key, value_str = key.strip(), value_part.strip().rstrip(';')
                try: raw_data[key] = ast.literal_eval(value_str)
                except (ValueError, SyntaxError): raw_data[key] = value_str
    except FileNotFoundError: print(f"错误: 数据文件未找到: {file_path}"); raise
    except Exception as e: print(f"读取文件时发生错误: {e}"); raise
    try:
        num_agents, num_jobs, num_machines = raw_data['g'], raw_data['job'], raw_data['machine']
        p_matrix_raw, e_matrix_raw = np.array(raw_data['P'], dtype=float), np.array(raw_data['E'], dtype=float)
        problem_def = ProblemDefinition(
            processing_times=p_matrix_raw.transpose(),
            power_consumption=e_matrix_raw.transpose(),
            release_times=np.array(raw_data['R'], dtype=int),
            agent_job_indices=np.array(raw_data['ag'], dtype=int),
            agent_weights=np.ones(num_agents, dtype=float),
            period_start_times=np.array(raw_data['U'], dtype=int),
            period_prices=np.array(raw_data['W'], dtype=float)
        )
        assert problem_def.num_agents == num_agents and problem_def.num_jobs == num_jobs and problem_def.num_machines == num_machines
    except (KeyError, ValueError) as e: print(f"数据格式错误或键缺失: {e}"); raise
    print(f"问题加载成功: {problem_def.num_jobs}个工件, {problem_def.num_machines}台机器, {problem_def.num_agents}个代理。")
    return problem_def

if __name__ == '__main__':
    data_dir = "dataset"
    file_path = os.path.join(data_dir, "data_A3_J400_M4_1.txt")

    try:
        problem = load_problem_from_file(file_path)
        print("\n--- 开始启发式搜索 ---")
        solver = HeuristicBBSolver(problem)
        print("\n[目标: TCTA]")
        solution_tcta = solver.find_heuristic_solution('TCTA', verbose=True)
        if solution_tcta:
            seq, pom = solution_tcta
            print(f"\n最终找到的序列 (TCTA): {seq}")
            print(f"最终的推迟决策 (TCTA, 仅第一列有效):\n{pom}")
        # print("\n[目标: TEC]")
        # solution_tec = solver.find_heuristic_solution('TEC', verbose=True)
        # if solution_tec:
        #     seq, pom = solution_tec
        #     print(f"\n最终找到的序列 (TEC): {seq}")
        #     print(f"最终的推迟决策 (TEC, 仅第一列有效):\n{pom}")
    except Exception as e:
        print(f"\n主程序发生错误: {e}")