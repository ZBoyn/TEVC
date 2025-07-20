import numpy as np
from config import ProblemDefinition, Solution
from typing import List

class Initializer:
    """根据不同策略初始化解的种群"""
    def __init__(self, problem_definition: ProblemDefinition, pop_size: int):
        self.problem = problem_definition
        self.pop_size = pop_size
        
        self.problem.agent_priority = self._calculate_agent_priorities()
    
    def _calculate_agent_priorities(self) -> List[int]:
        """实现 Heuristic 1, Step 1: 计算代理优先级

        Returns:
            List[int]: 一个按“加权完工时间”升序排列的代理ID列表。
        """
        print("计算代理优先级...")
        agent_weighted_times = []
        for agent_idx in range(self.problem.num_agents):
            # 获取该代理的所有工件
            start_job = self.problem.agent_job_indices[agent_idx]
            end_job = self.problem.agent_job_indices[agent_idx + 1]
            agent_jobs = list(range(start_job, end_job))
            
            # 估算当前单个代理的TCA
            estimated_tca = self._estimate_agent_tca(agent_jobs, num_samples=10)
            
            weight = self.problem.agent_weights[agent_idx]
            agent_weighted_times.append((estimated_tca * weight, agent_idx))

        # 按照加权时间升序排序
        agent_weighted_times.sort(key=lambda x: x[0])
        sorted_agent_ids = [agent_id for _, agent_id in agent_weighted_times]
        print(f"代理优先级 (从高到低): {sorted_agent_ids}")
        return sorted_agent_ids
    
    def _estimate_agent_tca(self, job_subset: List[int], num_samples: int = 10) -> float:
        """
        估算单个代理的TCA, 考虑所有约束, 并多次采样求平均。
        
        Args:
            job_subset (List[int]): 属于该代理的工件ID列表。
            num_samples (int): 为平滑随机性而进行的采样次数。

        Returns:
            float: 该代理预估的平均TCA
        """
        if not job_subset:
            return 0.0
        
        total_tca = 0.0
        for _ in range(num_samples):
            # 随机打乱当前代理的工件顺序
            sequence = np.random.permutation(job_subset)
            
            # 初始化完工时间矩阵 (行:工件, 列:机器)
            completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
            job_to_seq_idx = {job_id: i for i, job_id in enumerate(sequence)} # 将工件ID映射到序列索引

            # 紧凑调度
            for j_idx, job_id in enumerate(sequence):
                for i in range(self.problem.num_machines):
                    proc_time = self.problem.processing_times[job_id, i]
                    
                    # 计算理论上的最早开始时间 来自前序工件的约束 (在当前机器i上)
                    prev_job_id = sequence[j_idx - 1] if j_idx > 0 else -1
                    est_from_prev_job = completion_times[prev_job_id, i] if prev_job_id != -1 else 0
                    
                    # 来自本工件的前道工序的约束 (在前一台机器i-1上)
                    est_from_prev_machine = completion_times[job_id, i - 1] if i > 0 else 0
                    
                    # 结合工件自身释放时间
                    earliest_start_time = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])

                    # 处理电价时段约束 找到EST所在的时段索引
                    period_idx = np.searchsorted(self.problem.period_start_times, earliest_start_time, side='right') - 1

                    # 获取该时段的结束时间 (注意U[K]是总时长)
                    period_end_time = self.problem.period_start_times[period_idx + 1]
                    
                    actual_start_time = 0
                    if earliest_start_time + proc_time <= period_end_time:
                        # Case 1: 可以在当前时段内完成
                        actual_start_time = earliest_start_time
                    else:
                        # Case 2: 无法完成，必须推迟到下一个时段的开始
                        actual_start_time = period_end_time
                    
                    completion_times[job_id, i] = actual_start_time + proc_time
            last_job_completion = completion_times[sequence[-1], -1]
            total_tca += last_job_completion

        return total_tca / num_samples

    def _generate_sequence_by_simulation(self, selection_logic: callable) -> List[int]:
        """事件驱动的序列生成器通用框架
        
        Args:
            selection_logic (callable): 一个函数，用于从可用工件中选择下一个工件

        Returns:
            List[int]: 生成的工件调度序列
        """
        t = 0.0
        unscheduled_jobs = set(range(self.problem.num_jobs))
        final_sequence = []
        
        while unscheduled_jobs:
            # 找到在当前时间 t 已释放且可用的工件
            available_jobs = [j for j in unscheduled_jobs if self.problem.release_times[j] <= t]
            if not available_jobs:
                # 当前时间没有可用的工件 那么快进到下一个工件的释放时间
                
                next_release_time = min(self.problem.release_times[j] for j in unscheduled_jobs)
                t = next_release_time
                continue # 回到循环开始 重新确定可用工件

            # 使用传入的选择逻辑选择下一个工件
            next_job = selection_logic(available_jobs, t)
            final_sequence.append(next_job)
            unscheduled_jobs.remove(next_job)
            
            t += self.problem.processing_times[next_job, 0]  # 生成序列只关心第一个机器的处理时间
        return final_sequence
    
    def generate_with_heuristic1(self) -> Solution:
        """Heuristic 1: Available dominant agent first rule"""
        agent_priority_map = {agent_id: priority for priority, agent_id in enumerate(self.problem.agent_priority)}

        def selection_logic_h1(available_jobs: List[int], current_time: float) -> int:
            # 根据代理优先级对可用工件进行排序（优先级越小越靠前）
            available_jobs.sort(key=lambda job_id: agent_priority_map[self.problem.job_to_agent_map[job_id]])
            return available_jobs[0] # 返回最高优先级的工件
        
        sequence = self._generate_sequence_by_simulation(selection_logic_h1)
        return Solution(sequence=np.array(sequence), put_off=np.zeros_like(self.problem.processing_times))
    
    def generate_with_heuristic2(self) -> Solution:
        """Heuristic 2: The cheapest period first rule"""
        agent_priority_map = {agent_id: priority for priority, agent_id in enumerate(self.problem.agent_priority)}
        
        def selection_logic_h2(available_jobs: List[int], current_time: float) -> int:
            # 判断当前时间是否处于最低电价时段
            current_period = np.searchsorted(self.problem.period_start_times, current_time, side='right') - 1
            
            if current_period == self.problem.cheapest_period_index:
                # 如果是，按总能耗降序排序（能耗最高的优先）
                available_jobs.sort(key=lambda job_id: self.problem.job_total_energy[job_id], reverse=True)
            else:
                # 如果不是，按代理优先级升序排序
                available_jobs.sort(key=lambda job_id: agent_priority_map[self.problem.job_to_agent_map[job_id]])

            return available_jobs[0]

        sequence = self._generate_sequence_by_simulation(selection_logic_h2)
        return Solution(sequence=np.array(sequence), put_off=np.zeros_like(self.problem.processing_times))

    def generate_randomly(self) -> Solution:
        """随机生成一个个体"""
        sequence = np.random.permutation(self.problem.num_jobs)
        return Solution(sequence=sequence, put_off=np.zeros_like(self.problem.processing_times))

    def initialize_population(self, h1_count=1, h2_count=1, mutation_swaps=30) -> List[Solution]:
        """初始化种群

        Args:
            h1_count (int, optional): 使用启发式1生成的个体数量. Defaults to 1.
            h2_count (int, optional): 使用启发式2生成的个体数量. Defaults to 1.
            mutation_swaps (int, optional): 随机交换的次数. Defaults to 30.

        Returns:
            List[Solution]: 初始化后的种群
        """
        print("初始化种群...")
        if h1_count + h2_count > self.pop_size:
            raise ValueError("启发式个体总数不能超过种群大小。")
        
        population = []
        elites = []
        
        print(f"使用启发式1生成 {h1_count} 个个体...")
        for _ in range(h1_count):
            elite = self.generate_with_heuristic1()
            population.append(elite)
            elites.append(elite)
        
        print(f"使用启发式2生成 {h2_count} 个个体...")
        for _ in range(h2_count):
            elite = self.generate_with_heuristic2()
            population.append(elite)
            elites.append(elite)
        
        current_pop_size = len(population)
        # 循环生成剩余的个体
        for p in range(current_pop_size, self.pop_size):
            if p % 3 != 2:
                if elites: # 确保精英池不为空
                    # 对其序列进行多次交换操作
                    sequence_to_mutate = template_solution.sequence
                    for _ in range(mutation_swaps):
                        idx1, idx2 = np.random.choice(len(sequence_to_mutate), size=2, replace=False)
                        sequence_to_mutate[idx1], sequence_to_mutate[idx2] = sequence_to_mutate[idx2], sequence_to_mutate[idx1]
                    
                    population.append(template_solution)
                else: # 如果没有精英，则退化为随机生成
                    population.append(self.generate_randomly())
                template_solution = elites[np.random.randint(0, 2)].copy()

            else:
                # 每3个个体生成一个随机个体
                population.append(self.generate_randomly())
            
        print("种群初始化完成 共计:", len(population), "个体")
        return population