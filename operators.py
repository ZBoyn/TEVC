import numpy as np
from config import ProblemDefinition, Solution
from moea_tools import is_dominated
from typing import List

class BFO_Operators:
    """
    封装BFO核心算子工具
    包括趋向、复制、迁徙等操作
    被主算法调用
    """
    def __init__(self, problem_def: ProblemDefinition, decoder, bfo_params: dict):
        self.problem = problem_def
        self.decoder = decoder
        self.params = bfo_params
        
    def chemotaxis(self, parent_solution: Solution, current_step_size: float) -> Solution:
        """
        趋向操作: 以一个父代解为起点, 进行Mmax步的局部搜索, 返回最终找到的解

        Args:
            parent_solution (Solution): 父代解
            current_step_size (float): 当前步长

        Returns:
            Solution: 最终找到的解
        """
        m_max = self.params.get('Mmax', 10)
        put_off_mutation_prob = self.params.get('put_off_mutation_prob', 0.1)
        put_off_mutation_strength = self.params.get('put_off_mutation_strength', 1) # 每次变异的元素数量

        current_sol = parent_solution.copy()
        
        # BFO的位置编码, 仅在此算子内部临时使用
        positions = np.random.uniform(-1, 1, size=(self.problem.num_jobs, 2))
        
        for _ in range(m_max):
            original_sol = current_sol.copy()
            
            # 1. 生成一个候选解
            # 1a. 翻滚与游泳, 生成新序列
            theta = np.random.uniform(0, 2 * np.pi, self.problem.num_jobs)
            deta = np.column_stack([np.cos(theta), np.sin(theta)])
            positions += current_step_size * deta
            distances = np.linalg.norm(positions, axis=1)
            new_sequence = np.argsort(distances)
            
            # # 1b. 有偏向性地扰动put_off矩阵
            # new_put_off = current_sol.put_off.copy()
            # if np.random.rand() < put_off_mutation_prob:
            #     for _ in range(put_off_mutation_strength):
            #         job_idx = np.random.randint(self.problem.num_jobs)
            #         machine_idx = np.random.randint(self.problem.num_machines)
                    
            #         if new_put_off[job_idx, machine_idx] > 0 and np.random.rand() < 0.7: # 70%概率向0回归
            #             new_put_off[job_idx, machine_idx] -= 1
            #         else:
            #             new_put_off[job_idx, machine_idx] += 1
                        
            # 1c. 构造并评估候选解
            # trial_sol = Solution(sequence=new_sequence, put_off=new_put_off)
            
            # 1c. 构造并评估候选解 (put_off始终为0)
            trial_sol = Solution(sequence=new_sequence, put_off=np.zeros_like(original_sol.put_off))
            self.decoder.decode(trial_sol)

            # 2. 基于Pareto支配关系决定是否接受移动
            if is_dominated(trial_sol, current_sol):
                # 如果新解支配当前解, 则接受新解
                current_sol = trial_sol
            else:
                positions -= current_step_size * deta
            
        return current_sol
    
    def reproduction_crossover(self, parent1: Solution, parent2: Solution) -> tuple:
        """复制/交叉操作
            思想是让优秀个体(父代)繁殖产生后代(子代)

        Args:
            parent1 (Solution): _parent1
            parent2 (Solution): _parent2

        Returns:
            tuple: 结合两个父代的基因生成的两个子代
        """
        # 对工件序列sequence执行顺序交叉
        seq1, seq2 = parent1.sequence, parent2.sequence
        start, end = sorted(np.random.choice(len(seq1), 2, replace=False))
        
        child1_seq = -np.ones_like(seq1)
        child2_seq = -np.ones_like(seq2)
        
        # 复制中间段
        child1_seq[start:end+1] = seq1[start:end+1]
        child2_seq[start:end+1] = seq2[start:end+1]
        
        # 填充剩余部分
        p2_remaining = [item for item in seq2 if item not in child1_seq]
        p1_remaining = [item for item in seq1 if item not in child2_seq]
        
        # 从断点后开始填充
        for i in range(len(seq1)):
            idx = (end + 1 + i) % len(seq1)
            if child1_seq[idx] == -1: child1_seq[idx] = p2_remaining.pop(0)
            if child2_seq[idx] == -1: child2_seq[idx] = p1_remaining.pop(0)
        
        # 对putoff执行均匀交叉
        # put_off1, put_off2 = parent1.put_off, parent2.put_off
        # mask = np.random.rand(*put_off1.shape) < 0.5
        
        # child1_put_off = np.where(mask, put_off1, put_off2)
        # child2_put_off = np.where(mask, put_off2, put_off1)

        child1_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        child2_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        
        child1 = Solution(sequence=child1_seq, put_off=child1_put_off)
        child2 = Solution(sequence=child2_seq, put_off=child2_put_off)
        return child1, child2
    
    def migration(self, population: List[Solution]) -> List[Solution]:
        """
        自适应迁徙操作: 根据整个种群的表现，概率性地重置一部分"差"的个体

        Args:
            population (List[Solution]): 当前种群

        Returns:
            List[Solution]: 经过迁徙后的新种群
        """
        migrated_offspring = []
    
        # 提取所有个体的目标值 用于计算mc
        all_objectives = np.array([sol.objectives for sol in population])
        tec_values = all_objectives[:, 0]
        tcta_values = all_objectives[:, 1]
        
        # 计算每个个体的"差度" 越高表示越差
        epsilon = 1e-9
        tec_range = tec_values.max() - tec_values.min() + epsilon
        tcta_range = tcta_values.max() - tcta_values.min() + epsilon
        
        norm_tec = (tec_values - tec_values.min()) / tec_range
        norm_tcta = (tcta_values - tcta_values.min()) / tcta_range
        
        mc = 0.5 * norm_tec + 0.5 * norm_tcta
        
        # 遍历种群 根据概率决定哪些个体需要迁徙
        for i, sol in enumerate(population):
            if np.random.rand() < mc[i]:
                # 触发迁徙 生成一个全新随机解
                new_sequence = np.random.permutation(self.problem.num_jobs)
                new_put_off = np.zeros_like(population[i].put_off)
                migrated_offspring.append(Solution(sequence=new_sequence, put_off=new_put_off))
            else:
                # 未触发 直接保留父代
                migrated_offspring.append(population[i].copy())
        
        return migrated_offspring
    
class LocalSearch_Operators:
    """
    本地搜索算子 用于在当前解附近进行局部优化
    1. 优势代理
    2. 右移优化
    """
    def __init__(self, problem_def: ProblemDefinition, decoder):
        self.problem = problem_def
        self.decoder = decoder

    def prefer_agent(self, parent_solution: Solution) -> Solution:
        """优势代理局部搜索算子

        Args:
            parent_solution (Solution): 父代解

        Returns:
            Solution: 经过优势代理优化后的解
        """
        child_solution = parent_solution.copy()
        sequence = child_solution.sequence
        
        highest_priority_agent_id = self.problem.agent_priority[0]
    
        left, right = 0, self.problem.num_jobs - 1
        while left < right:
            # 从右向左, 找到第一个属于最高优先级代理的工件
            is_right_job_dominant = (self.problem.job_to_agent_map.get(sequence[right]) == highest_priority_agent_id)
            if not is_right_job_dominant:
                right -= 1
                continue
            # 从左向右, 找到第一个不属于最高优先级代理的工件
            is_left_job_dominant = (self.problem.job_to_agent_map.get(sequence[left]) == highest_priority_agent_id)
            if is_left_job_dominant:
                left += 1
                continue
            
            # 如果左右指针都找到了目标且还未相遇, 则交换
            if left < right:
                sequence[left], sequence[right] = sequence[right], sequence[left]
                left += 1
                right -= 1
                
        child_solution.sequence = sequence
        
        child_solution.put_off = np.zeros_like(parent_solution.put_off)
        
        return child_solution

    def right_shift(self, parent_solution: Solution) -> Solution:
        child_solution = parent_solution.copy()
        sequence = child_solution.sequence
        
        # 阶段一: 正向传播 计算最早时间
        earliest_completion_times = self._calculate_earliest_times(sequence)
        
        # 阶段二: 反向传播 计算最晚时间
        latest_completion_times = self._calculate_latest_times(sequence, earliest_completion_times)

        # 阶段三: 决策 生成新的put_off矩阵
        new_put_off = np.zeros_like(child_solution.put_off)
        final_completion_times = earliest_completion_times.copy()
        
        # 必须按逆序进行决策, 以确保依赖关系正确
        for job_id in reversed(sequence):
            for machine_id in range(self.problem.num_machines - 1, -1, -1):
                proc_time = self.problem.processing_times[job_id, machine_id]
                
                # a. 确定此操作的最早和最晚开始时间
                earliest_start_time = earliest_completion_times[job_id, machine_id] - proc_time
                latest_start_time = latest_completion_times[job_id, machine_id] - proc_time
                
                if latest_start_time < earliest_start_time: continue # 没有移动空间

                # b. 找到此时间窗口[EST, LST]覆盖的所有电价时段
                start_period = np.searchsorted(self.problem.period_start_times, earliest_start_time, side='right') - 1
                end_period = np.searchsorted(self.problem.period_start_times, latest_start_time, side='right') - 1
                
                # c. 在可行时段中, 选择最便宜的一个
                candidate_periods = list(range(start_period, end_period + 1))
                if not candidate_periods: continue
                
                cheapest_period_idx = min(candidate_periods, key=lambda p_idx: self.problem.period_prices[p_idx])
                
                # d. 尝试将操作安排在最便宜时段的开始
                target_start_time = self.problem.period_start_times[cheapest_period_idx]
                
                # 新的开始时间必须 >= 最早开始时间, 且 <= 最晚开始时间
                new_start_time = max(earliest_start_time, target_start_time)
                
                # 检查新的完工时间是否会超过最晚完工时间
                if new_start_time + proc_time > latest_completion_times[job_id, machine_id] + 1e-6:
                    # 如果移动到时段开头会超时，则保持原样（紧凑排列）
                    new_start_time = earliest_start_time

                # e. 计算并存储所需的put_off值
                required_delay = new_start_time - earliest_start_time
                new_put_off[job_id, machine_id] = required_delay
        
        # 将put_off从"时间推迟"转换为我们的"时段推迟"编码
        final_put_off_periods = np.zeros_like(new_put_off, dtype=int)
        for job_id in range(self.problem.num_jobs):
             for machine_id in range(self.problem.num_machines):
                est = earliest_completion_times[job_id, machine_id] - self.problem.processing_times[job_id, machine_id]
                delayed_est = est + new_put_off[job_id, machine_id]
                
                base_period = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                final_period = np.searchsorted(self.problem.period_start_times, delayed_est, side='right') - 1
                final_put_off_periods[job_id, machine_id] = max(0, final_period - base_period)

        child_solution.put_off = final_put_off_periods
        return child_solution
    
    def _calculate_earliest_times(self, sequence: np.ndarray) -> np.ndarray:
        """辅助函数: 执行正向传播, 计算紧凑调度下的完工时间

        Args:
            sequence (np.ndarray): 工件序列

        Returns:
            np.ndarray: 每个工件在每台机器上的最早完工时间
        """
        # 逻辑和Deocoder.decode()类似(put_off矩阵为0)
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        for j_idx, job_id in enumerate(sequence):
            for i in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, i]
                prev_job_id = sequence[j_idx - 1] if j_idx > 0 else -1
                est_from_prev_job = completion_times[prev_job_id, i] if prev_job_id != -1 else 0
                est_from_prev_machine = completion_times[job_id, i - 1] if i > 0 else 0
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])
                
                period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                period_end_time = self.problem.period_start_times[period_idx + 1]
                start_time = est if est + proc_time <= period_end_time else period_end_time
                completion_times[job_id, i] = start_time + proc_time
        return completion_times
    
    def _calculate_latest_times(self, sequence: np.ndarray, earliest_times: np.ndarray) -> np.ndarray:
        """辅助函数: 执行反向传播, 计算最晚完工时间

        Args:
            sequence (np.ndarray): 工件序列
            earliest_times (np.ndarray): 每个工件在每台机器上的最早完工时间

        Returns:
            np.ndarray: 每个工件在每台机器上的最晚完工时间
        """
        latest_completion = np.full((self.problem.num_jobs, self.problem.num_machines), np.inf)
        
        # 使用最早完工时间的最大值作为截止日期
        deadline = earliest_times.max()
        latest_completion[sequence[-1], -1] = deadline # 最后一个工件在最后一台机器上的最晚完工时间
        
        for i in range(self.problem.num_machines - 1, -1, -1):
            for j_idx in range(self.problem.num_jobs - 1, -1, -1):
                job_id = sequence[j_idx]
                proc_time = self.problem.processing_times[job_id, i]
                
                # 来自后序工件的约束
                next_job_id = sequence[j_idx + 1] if j_idx < self.problem.num_jobs - 1 else -1
                limit_from_next_job = latest_completion[next_job_id, i] - self.problem.processing_times[next_job_id, i] if next_job_id != -1 else deadline
                
                # 来自本工件的后道工序的约束
                limit_from_next_machine = latest_completion[job_id, i + 1] - self.problem.processing_times[job_id, i+1] if i < self.problem.num_machines - 1 else deadline
                
                latest_completion[job_id, i] = min(limit_from_next_job, limit_from_next_machine)
        
        return latest_completion