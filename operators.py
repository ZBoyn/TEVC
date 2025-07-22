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
        """
        右移策略 (Right Shift) - 新的稳健实现
        目标: 给定一个固定的工件序列, 在不显著恶化完工时间的前提下,
               通过推迟非关键路径上的工序来最小化总电力成本 (TEC).
        """
        # 0. 准备工作
        child_solution = parent_solution.copy()
        sequence = child_solution.sequence
        
        # 确保输入解是紧凑的 (put_off为0)
        child_solution.put_off = np.zeros_like(parent_solution.put_off)
        self.decoder.decode(child_solution)
        
        # 1. 正向传播: 计算最早开始/完成时间 (EST / ECT)
        ect_matrix = self._calculate_earliest_times(sequence)
        
        # 2. 反向传播: 计算最晚开始/完成时间 (LST / LCT)
        lct_matrix = self._calculate_latest_times(sequence, ect_matrix)

        # 3. 决策与移动: 直接计算最优的 put_off 时段数
        final_put_off_periods = np.zeros_like(child_solution.put_off, dtype=int)

        for job_id in sequence:
            for machine_id in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, machine_id]
                
                est = ect_matrix[job_id, machine_id] - proc_time
                lst = lct_matrix[job_id, machine_id] - proc_time
                
                if lst - est < 1e-6:
                    continue

                base_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                
                cheapest_price = np.inf
                best_target_period_idx = base_period_idx

                # 寻找最优目标时段
                # 遍历所有 est 所在的时段 到 lst 可能结束的时段
                start_p_idx = base_period_idx
                end_p_idx = np.searchsorted(self.problem.period_start_times, lst, side='right') - 1
                
                current_price = self.problem.period_prices[base_period_idx]

                for p_idx in range(start_p_idx, end_p_idx + 2):
                    if p_idx >= len(self.problem.period_prices): continue
                    
                    # 模拟decode的行为, 计算移动到该时段后的实际开始时间
                    potential_start_time = max(est, self.problem.period_start_times[p_idx])
                    
                    if potential_start_time > lst + 1e-6:
                        break # 超出最晚开始时间, 后续时段更不可行
                    
                    price = self.problem.period_prices[p_idx]
                    if price < current_price:
                        current_price = price
                        best_target_period_idx = p_idx
                
                final_put_off_periods[job_id, machine_id] = max(0, best_target_period_idx - base_period_idx)

        child_solution.put_off = final_put_off_periods
        #【重要】应用了新的put_off后,需要重新解码一次以更新目标函数值
        self.decoder.decode(child_solution)
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
        
        # 使用问题的全局deadline作为反向传播的起点
        deadline = self.problem.deadline
        
        # 初始化所有无后继的操作的最晚完成时间为deadline
        last_job_in_sequence = sequence[-1]
        for m_id in range(self.problem.num_machines):
            latest_completion[last_job_in_sequence, m_id] = deadline
        # 修复: ProblemDefinition没有job_ids属性, 使用num_jobs生成ID.
        for j_id in range(self.problem.num_jobs):
            latest_completion[j_id, self.problem.num_machines - 1] = deadline


        for i in range(self.problem.num_machines - 1, -1, -1):
            # 需要按序列的逆序进行
            for j_idx in range(self.problem.num_jobs - 1, -1, -1):
                job_id = sequence[j_idx]
                
                # 来自后序工件的约束 (同一台机器)
                limit_from_next_job_on_machine = np.inf
                if j_idx < self.problem.num_jobs - 1:
                    next_job_id = sequence[j_idx + 1]
                    limit_from_next_job_on_machine = latest_completion[next_job_id, i] - self.problem.processing_times[next_job_id, i]
                
                # 来自本工件的后道工序的约束
                limit_from_next_machine_for_job = np.inf
                if i < self.problem.num_machines - 1:
                    limit_from_next_machine_for_job = latest_completion[job_id, i + 1] - self.problem.processing_times[job_id, i + 1]

                current_lct = min(latest_completion[job_id, i], limit_from_next_job_on_machine, limit_from_next_machine_for_job)
                latest_completion[job_id, i] = current_lct
        
        return latest_completion