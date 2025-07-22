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
            
            # 1b. 有偏向性地扰动put_off矩阵
            new_put_off = current_sol.put_off.copy()
            if np.random.rand() < put_off_mutation_prob:
                for _ in range(put_off_mutation_strength):
                    job_idx = np.random.randint(self.problem.num_jobs)
                    machine_idx = np.random.randint(self.problem.num_machines)
                    
                    if new_put_off[job_idx, machine_idx] > 0 and np.random.rand() < 0.7: # 70%概率向0回归
                        new_put_off[job_idx, machine_idx] -= 1
                    else:
                        new_put_off[job_idx, machine_idx] += 1
                        
            # 1c. 构造并评估候选解 (final_schedule=None, put_off=0)
            trial_sol = Solution(sequence=new_sequence, put_off=np.zeros_like(original_sol.put_off), final_schedule=None)
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
        
        child1 = Solution(sequence=child1_seq, put_off=child1_put_off, final_schedule=None)
        child2 = Solution(sequence=child2_seq, put_off=child2_put_off, final_schedule=None)
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
                migrated_offspring.append(Solution(sequence=new_sequence, put_off=new_put_off, final_schedule=None))
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
        
        # 确保返回的解是一个标准的紧凑排程解
        child_solution.put_off = np.zeros_like(parent_solution.put_off)
        child_solution.final_schedule = None
        
        return child_solution

    def right_shift(self, parent_solution: Solution) -> Solution:
        """
        右移策略 (Right Shift) - 采用反向传播重写, 解决级联延迟问题
        """
        # 0. 准备工作
        child_solution = parent_solution.copy()
        sequence = child_solution.sequence
        num_jobs, num_machines = self.problem.num_jobs, self.problem.num_machines

        # 1. 计算基准的最早完成时间 (前向传播)
        # 这个矩阵是固定的, 作为计算任何工序EST的依据
        base_ect_matrix = self._calculate_earliest_times(sequence)

        # 2. 核心: 反向传播计算最终调度
        # final_ct_matrix 用于存储我们为每个工序确定的最终完成时间
        final_ct_matrix = base_ect_matrix.copy()

        for i in range(num_machines - 1, -1, -1):  # 从最后一道机器开始
            for j_idx in range(num_jobs - 1, -1, -1): # 从序列中最后一个工件开始
                job_id = sequence[j_idx]
                proc_time = self.problem.processing_times[job_id, i]

                # 2a. 确定当前工序(j,i)的调度时间窗 [est, lct]
                # EST (最早开始时间) 由其前序工序的 *基准* 完成时间决定
                est_from_prev_job = base_ect_matrix[sequence[j_idx - 1], i] if j_idx > 0 else 0
                est_from_prev_machine = base_ect_matrix[job_id, i - 1] if i > 0 else 0
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])

                # LCT (最晚完成时间) 由其后继工序 *已确定* 的最终开始时间决定
                lct_from_next_job = np.inf
                if j_idx < num_jobs - 1:
                    next_job_id = sequence[j_idx + 1]
                    # 后继工件的最终开始时间 = 最终完成时间 - 它的处理时间
                    lct_from_next_job = final_ct_matrix[next_job_id, i] - self.problem.processing_times[next_job_id, i]

                lct_from_next_machine = np.inf
                if i < num_machines - 1:
                    # 后继工序的最终开始时间
                    lct_from_next_machine = final_ct_matrix[job_id, i + 1] - self.problem.processing_times[job_id, i + 1]
                
                # 同时要受全局 Deadline 约束
                lct = min(lct_from_next_job, lct_from_next_machine, self.problem.deadline)
                lst = lct - proc_time

                # 如果没有可移动空间, 则该工序的最终完成时间就是其基准ECT, 直接进入下一次循环
                if lst - est < 1e-6:
                    final_ct_matrix[job_id, i] = base_ect_matrix[job_id, i]
                    continue

                # 2b. 在[est, lst]窗口内, 为当前工序寻找成本最低的开始时间
                best_start_time = est
                min_cost = self.decoder.calculate_op_cost(est, proc_time)

                # 遍历所有可能涉及到的时段
                start_p_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                end_p_idx = np.searchsorted(self.problem.period_start_times, lst, side='right') - 1

                for p_idx in range(start_p_idx, end_p_idx + 2):
                    if p_idx >= len(self.problem.period_prices): continue
                    
                    potential_start = max(est, self.problem.period_start_times[p_idx])
                    
                    # 模拟解码器, 计算真实开始时间 (处理非抢占)
                    p_end = self.problem.period_start_times[p_idx + 1]
                    actual_start_time = potential_start if potential_start + proc_time <= p_end else p_end

                    if actual_start_time > lst + 1e-6:
                        break # 后续的移动更不可行

                    cost = self.decoder.calculate_op_cost(actual_start_time, proc_time)
                    if cost < min_cost:
                        min_cost = cost
                        best_start_time = actual_start_time

                # 2c. 根据找到的最佳开始时间, 更新该工序的最终完成时间
                final_ct_matrix[job_id, i] = best_start_time + proc_time

        # 3. 后处理: 将计算出的最终调度结果直接存入解中
        child_solution.final_schedule = final_ct_matrix
        child_solution.put_off = np.zeros_like(child_solution.put_off) # put_off不再使用, 但保持结构完整

        # 最后用标准的解码器验证并获得最终目标值
        self.decoder.decode(child_solution)
        return child_solution

    def destroy_rebuild(self, parent_solution: Solution, alpha: float) -> Solution:
        """
        基于NEH思想的破坏与重建算子.
        1. 破坏: 移除一部分加工时间最短的"灵活"工件.
        2. 重建: 按照总加工时间最长(LPT)的顺序, 将被移除的工件逐个插入到最佳位置.
        """
        child_solution = parent_solution.copy()
        initial_sequence = child_solution.sequence
        num_jobs = self.problem.num_jobs

        # 1. 破坏阶段
        # 直接从problem definition中获取预先计算好的信息
        total_processing_times = self.problem.job_total_processing_times
        sorted_jobs_by_proc_time_asc = self.problem.job_spt_order
        
        num_to_destroy = int(alpha * num_jobs)
        if num_to_destroy == 0:
            return child_solution # 如果破坏数量为0, 直接返回原解

        jobs_to_destroy = set(sorted_jobs_by_proc_time_asc[:num_to_destroy])
        
        # 得到被破坏后的部分序列
        partial_sequence = [job for job in initial_sequence if job not in jobs_to_destroy]

        # 2. 重建阶段
        # 按总加工时间降序(LPT)排序待插入的工件
        jobs_to_rebuild = sorted(list(jobs_to_destroy), key=lambda k: -total_processing_times[k])

        current_sequence = partial_sequence
        for job_to_insert in jobs_to_rebuild:
            best_insertion_sequence = None
            min_tcta = float('inf')
            
            # 尝试所有可能的插入位置
            for i in range(len(current_sequence) + 1):
                temp_sequence_list = current_sequence[:i] + [job_to_insert] + current_sequence[i:]
                
                # 构造临时解并评估 (put_off为0, 专注于序列优化)
                temp_sol = Solution(sequence=np.array(temp_sequence_list), put_off=np.zeros_like(child_solution.put_off))
                self.decoder.decode(temp_sol)
                
                # 评价标准: 选择使 TCTA 最小的插入位置
                current_tcta = temp_sol.objectives[1] # TCTA 在第二个目标
                if current_tcta < min_tcta:
                    min_tcta = current_tcta
                    best_insertion_sequence = temp_sequence_list
            
            # 【修复】如果所有插入位置都导致无效解, best_insertion_sequence会是None.
            # 这种情况下, 我们选择一个默认行为(例如插入到末尾)来防止程序崩溃.
            # 产生的无效解会被后续的进化过程自然淘汰.
            if best_insertion_sequence is None:
                current_sequence = current_sequence + [job_to_insert]
            else:
                current_sequence = best_insertion_sequence
            
        # 3. 返回最终解 (标准的紧凑排程解)
        final_solution = Solution(sequence=np.array(current_sequence), 
                                  put_off=np.zeros_like(child_solution.put_off),
                                  final_schedule=None)
        return final_solution

    def _calculate_earliest_times(self, sequence: np.ndarray) -> np.ndarray:
        """
        辅助函数: 执行正向传播, 计算紧凑调度下的完工时间

        Args:
            sequence (np.ndarray): 工件序列

        Returns:
            np.ndarray: 每个工件在每台机器上的最早完工时间
        """
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        
        for j_idx, job_id in enumerate(sequence):
            for i in range(self.problem.num_machines):
                proc_time = self.problem.processing_times[job_id, i]
                prev_job_id = sequence[j_idx - 1] if j_idx > 0 else -1
                
                est_from_prev_job = completion_times[prev_job_id, i] if prev_job_id != -1 else 0
                est_from_prev_machine = completion_times[job_id, i - 1] if i > 0 else 0
                
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])
                
                # 确定EST所在的时段
                period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
                period_idx = max(0, period_idx) # 确保索引不为负

                start_time = est
                
                # 检查是否需要跨时段
                if period_idx < self.problem.num_periods - 1:
                    period_end_time = self.problem.period_start_times[period_idx + 1]
                    if start_time + proc_time > period_end_time:
                        start_time = period_end_time # 非抢占, 只能推到下一个时段开始
                
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
        # 使用问题的全局 deadline 作为反向传播的起点.
        # 这为 right_shift 提供了最大的可操作空间, 以便用时间换成本.
        deadline = self.problem.deadline
        
        latest_completion = np.full((self.problem.num_jobs, self.problem.num_machines), np.inf)
        
        # 初始化所有无后继的操作(即序列最后一个工件, 和所有工件的最后一道工序)的最晚完成时间为deadline
        last_job_in_sequence = sequence[-1]
        last_machine_id = self.problem.num_machines - 1
        
        for m_id in range(self.problem.num_machines):
            latest_completion[last_job_in_sequence, m_id] = deadline
        for j_id in range(self.problem.num_jobs):
            latest_completion[j_id, last_machine_id] = deadline
        
        # 确保最后一个操作的 LCT 就是 deadline
        latest_completion[last_job_in_sequence, last_machine_id] = deadline

        # 按机器和工件序列的逆序进行反向传播
        for i in range(self.problem.num_machines - 1, -1, -1):
            for j_idx in range(self.problem.num_jobs - 1, -1, -1):
                job_id = sequence[j_idx]
                
                # 来自后序工件的约束 (同一台机器)
                limit_from_next_job_on_machine = np.inf
                if j_idx < self.problem.num_jobs - 1:
                    next_job_id = sequence[j_idx + 1]
                    # LST of next job on the same machine
                    limit_from_next_job_on_machine = latest_completion[next_job_id, i] - self.problem.processing_times[next_job_id, i]
                
                # 来自本工件的后道工序的约束
                limit_from_next_machine_for_job = np.inf
                if i < self.problem.num_machines - 1:
                    # LST of same job on the next machine
                    limit_from_next_machine_for_job = latest_completion[job_id, i + 1] - self.problem.processing_times[job_id, i + 1]

                # 当前已有的 LCT (可能来自之前的初始化或更强的约束)
                current_lct_limit = latest_completion[job_id, i]
                
                latest_completion[job_id, i] = min(current_lct_limit, limit_from_next_job_on_machine, limit_from_next_machine_for_job)
        
        return latest_completion