import numpy as np
from pro_def import ProblemDefinition, Solution
from moea_tools import is_dominated
from typing import List, Dict
import os

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
        趋向操作: 以一个父代解为起点, 进行Mmax步的迭代局部搜索, 返回最终找到的最优解.
        这是一个迭代改进的过程, 每一步都在当前找到的最优解的基础上进行探索.
        """
        m_max = self.params.get('Mmax', 10)
        put_off_mutation_prob = self.params.get('put_off_mutation_prob', 0.1)
        put_off_mutation_strength = self.params.get('put_off_mutation_strength', 1)
        put_off_regression_prob = self.params.get('put_off_regression_prob', 0.7)

        # 1. 初始化: 当前最优解就是父代
        current_best_sol = parent_solution.copy()
        # 确保初始解已被评估
        if current_best_sol.objectives is None:
            self.decoder.decode(current_best_sol)
        
        # BFO的位置编码, 用于探索邻域
        positions = np.random.uniform(-1, 1, size=(self.problem.num_jobs, 2))
        
        # 2. 迭代搜索
        for _ in range(m_max):
            # 2a. 生成一个基于当前最优解的候选解
            # 翻滚与游泳, 生成新序列 (探索邻域)
            theta = np.random.uniform(0, 2 * np.pi, self.problem.num_jobs)
            deta = np.column_stack([np.cos(theta), np.sin(theta)])
            new_positions = positions + current_step_size * deta
            
            distances = np.linalg.norm(new_positions, axis=1)
            new_sequence = np.argsort(distances)
            
                        
            # 有偏向性地扰动put_off矩阵
            new_put_off = current_best_sol.put_off.copy() # 从当前最优解的put_off开始
            if np.random.rand() < put_off_mutation_prob:
                for _ in range(put_off_mutation_strength):
                    job_idx = np.random.randint(self.problem.num_jobs)
                    machine_idx = np.random.randint(self.problem.num_machines)
                    
                    if new_put_off[job_idx, machine_idx] > 0 and np.random.rand() < put_off_regression_prob:
                        new_put_off[job_idx, machine_idx] -= 1
                    else:
                        new_put_off[job_idx, machine_idx] += 1
                        
            # 构造并评估候选解
            trial_sol = Solution(sequence=new_sequence, put_off=new_put_off, final_schedule=None)
            self.decoder.decode(trial_sol)

            # 2b. 基于Pareto支配关系决定是否接受移动
            # 如果候选解支配当前最优解, 则更新最优解和探索位置
            if is_dominated(trial_sol, current_best_sol):
                current_best_sol = trial_sol
                positions = new_positions # 接受移动, 更新探索的中心点
            # 否则, 不更新, 留在原地, positions不变, 在下一次循环从原位置继续探索
            
        return current_best_sol
    
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
        put_off1, put_off2 = parent1.put_off, parent2.put_off
        mask = np.random.rand(*put_off1.shape) < 0.5
        
        child1_put_off = np.where(mask, put_off1, put_off2)
        child2_put_off = np.where(mask, put_off2, put_off1)

        # child1_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        # child2_put_off = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        
        child1 = Solution(sequence=child1_seq, put_off=child1_put_off, final_schedule=None)
        child2 = Solution(sequence=child2_seq, put_off=child2_put_off, final_schedule=None)
        return child1, child2
    
    def migration(self, population: List[Solution]) -> List[Solution]:
        """
        自适应迁徙操作: 根据整个种群的表现,概率性地重置一部分"差"的个体

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
        
        tec_weight = self.params.get('migration_tec_weight', 0.5)
        tcta_weight = self.params.get('migration_tcta_weight', 0.5)
        mc = tec_weight * norm_tec + tcta_weight * norm_tcta
        
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

    def _first_valid_start(self, est: float, proc_time: float) -> float:
        """给定最早可能开始时间 est, 返回第一个能够在同一时段内加工完毕的开始时间。
        若工序处理时间大于单时段长度, 会持续顺延到能够完整容纳的时段起点。
        """
        period_starts = self.problem.period_start_times
        num_periods = self.problem.num_periods

        cur_start = est
        while True:
            period_idx = np.searchsorted(period_starts, cur_start, side='right') - 1
            # 如果已经超出最后一个已定义时段, 直接返回 (后续调度器会视为不合法解)
            if period_idx >= num_periods - 1:
                return cur_start
            period_end = period_starts[period_idx + 1]
            if cur_start + proc_time <= period_end:
                return cur_start  # 本时段可完成
            # 否则推到下一时段起点
            cur_start = period_end

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

    def _get_valid_time_slots(self, est: float, lst: float, proc_time: float) -> List[Dict]:
        """
        分析[est, lst]时间窗, 结合电价时段, 返回所有有效的、可放置工序的连续时间间隙 (时隙).
        新版逻辑: 遍历所有时段, 计算交集, 确保稳健性.
        """
        if est > lst + 1e-6:
            return []
            
        slots = []
        
        # 遍历每一个电价时段, 寻找交叉点
        for p_idx in range(self.problem.num_periods):
            period_start = self.problem.period_start_times[p_idx]
            period_end = self.problem.period_start_times[p_idx + 1] if p_idx + 1 < len(self.problem.period_start_times) else self.problem.deadline
            
            # 1. 确定在本时段内, 工序的开始时间可以存在的有效范围
            # 一个工序要完全包含在本时段, 其开始时间必须在 [period_start, period_end - proc_time]
            period_valid_start = period_start
            period_valid_end = period_end - proc_time

            # 2. 计算这个范围与工序自身约束[est, lst]的交集
            intersection_start = max(est, period_valid_start)
            intersection_end = min(lst, period_valid_end)

            # 3. 如果交集存在, 则它是一个有效的时隙
            if intersection_start <= intersection_end + 1e-6:
                slots.append({
                    'start': intersection_start,
                    'end': intersection_end,
                    'price': self.problem.period_prices[p_idx]
                })
        
        return slots


    def _find_best_start_time(self, est: float, lst: float, proc_time: float, is_last_machine: bool, is_agent_last_job: bool) -> float:
        """
        在[est, lst]窗口内选择最佳开始时间.
        
        策略:
        1. 非最后一台机器的工件: 
           - 在不增大TEC且满足约束的基础上右移到LST(相等也右移)
           - 为后续工件腾出空间
        2. 最后一台机器的工件: 
           - 代理最后工件: 同时段用EST,跨时段且更便宜可移动
           - 非代理最后工件: 同时段用LST,跨时段且更便宜可移动
        
        硬约束: 不允许工件跨越时段加工
        """
        # 1. 获取所有有效的时隙(已确保不跨时段)
        valid_slots = self._get_valid_time_slots(est, lst, proc_time)

        # 如果没有有效的时隙,意味着在[est, lst]内无法安放该工序而不跨越时段.
        # 这种情况理论上是算法实现有问题,但作为保护,我们返回一个修正过的est,
        # 尽力满足约束,尽管可能不是最优.
        if not valid_slots:
            # 修正est使其不跨时段
            corrected_est = self._first_valid_start(est, proc_time)
            # 确保修正后的时间点不晚于lst
            return min(corrected_est, lst)
            
        if not is_last_machine:
            # 非最后一台机器: 在不增大TEC的基础上右移到LST
            # 找到当前EST所在时段的电价
            est_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
            current_price = self.problem.period_prices[est_period_idx]
            
            # 检查LST所在时隙的电价
            lst_slot = None
            for slot in valid_slots:
                # 必须是 <= lst, 因为lst本身就是一个有效的可开始时间点
                if slot['start'] <= lst <= slot['end'] + 1e-6:
                    lst_slot = slot
                    break
            
            if lst_slot and lst_slot['price'] <= current_price:
                # LST所在时隙的电价可接受, 直接返回LST
                return lst
            else:
                # LST所在时隙太贵, 找最后一个负担得起的时隙
                affordable_slots = [s for s in valid_slots if s['price'] <= current_price]
                if affordable_slots:
                    return affordable_slots[-1]['end']
                else:
                    # 如果所有时隙都比当前更贵, 留在EST, 但需保证EST有效
                    return min(self._first_valid_start(est, proc_time), lst)
        else:
            # 最后一台机器: 按原策略执行
            # 检查EST和LST是否在同一时段
            est_period_idx = np.searchsorted(self.problem.period_start_times, est, side='right') - 1
            lst_period_idx = np.searchsorted(self.problem.period_start_times, lst, side='right') - 1
            same_period = (est_period_idx == lst_period_idx)
            
            # 找到最低电价
            min_price = min(slot['price'] for slot in valid_slots)
            cheapest_slots = [s for s in valid_slots if abs(s['price'] - min_price) < 1e-6]
            
            if same_period:
                # 同时段情况
                if is_agent_last_job:
                    # 代理最后工件: 使用EST(不影响TCTA), 但需保证其有效性
                    return min(self._first_valid_start(est, proc_time), lst)
                else:
                    # 非代理最后工件: 使用最晚可用时间(为后续工件腾空间)
                    # 直接返回 lst 是错误的, 因为 lst 可能导致跨时段.
                    # 正确做法是返回当前时段内有效的最晚开始时间.
                    # 因为 same_period 为 True, valid_slots 中只有一个时隙.
                    return valid_slots[0]['end']
            else:
                # 跨时段情况: 两种工件都移动到最便宜时段
                if is_agent_last_job:
                    # 代理最后工件: 移动到最便宜时段的开始位置
                    return cheapest_slots[0]['start']
                else:
                    # 非代理最后工件: 移动到最便宜时段内的最晚位置
                    target_slot = cheapest_slots[-1]  # 最后一个最便宜时隙
                    return target_slot['end']

    def right_shift(self, parent_solution: Solution) -> Solution:
        """
        右移策略 (Right Shift) - 采用反向传播重写, 解决级联延迟问题
        新版逻辑: 根据工序是否为代理的最终考核点, 以及是否在最终机器上, 采用不同推迟策略.
        """
        child_solution = parent_solution.copy()
        sequence = child_solution.sequence
        num_jobs, num_machines = self.problem.num_jobs, self.problem.num_machines

        

        # 1. 计算基准的最早完成时间 (前向传播)
        base_ect_matrix = self._calculate_earliest_times(sequence)

        # 2. 核心: 反向传播计算最终调度
        final_ct_matrix = base_ect_matrix.copy()
        
        # 在最后一台机器上, 需要判断是否为代理的最后一个工件
        seen_agents_on_last_machine = set()

        for i in range(num_machines - 1, -1, -1):  # 从最后一道机器开始
            is_last_machine = (i == num_machines - 1)

            for j_idx in range(num_jobs - 1, -1, -1): # 从序列中最后一个工件开始
                job_id = sequence[j_idx]
                proc_time = self.problem.processing_times[job_id, i]

                # 2a. 确定当前工序(j,i)的调度时间窗 [est, lct]
                # 前一个工件在反向遍历中还未处理,使用base_ect_matrix
                est_from_prev_job = base_ect_matrix[sequence[j_idx - 1], i] if j_idx > 0 else 0
                # 前一台机器在反向遍历中还未处理,使用base_ect_matrix
                est_from_prev_machine = base_ect_matrix[job_id, i - 1] if i > 0 else 0
                est = max(est_from_prev_job, est_from_prev_machine, self.problem.release_times[job_id])

                lct_from_next_job = np.inf
                if j_idx < num_jobs - 1:
                    next_job_id = sequence[j_idx + 1]
                    lct_from_next_job = final_ct_matrix[next_job_id, i] - self.problem.processing_times[next_job_id, i]

                lct_from_next_machine = np.inf
                if i < num_machines - 1:
                    lct_from_next_machine = final_ct_matrix[job_id, i + 1] - self.problem.processing_times[job_id, i + 1]
                
                lct = min(lct_from_next_job, lct_from_next_machine, self.problem.deadline)
                lst = lct - proc_time

                # 如果没有可移动空间, 则该工序的最终完成时间就是其基准ECT, 直接进入下一次循环
                if lst - est < 1e-6:
                    # 在最后一台机器上,即使没有移动空间,也要维护seen_agents状态
                    if is_last_machine:
                        agent_id = self.problem.job_to_agent_map[job_id]
                        seen_agents_on_last_machine.add(agent_id)

                        
                    final_ct_matrix[job_id, i] = base_ect_matrix[job_id, i]
                    continue
                
                # 2b. 根据新策略寻找最佳开始时间
                is_agent_last_job = False # 默认不是代理最后工件
                if is_last_machine:
                    agent_id = self.problem.job_to_agent_map[job_id]
                    
                    # 由于是反向遍历, 第一次遇到的代理工件就是该代理的最终考核点
                    is_last_job_for_agent = (agent_id not in seen_agents_on_last_machine)

                    if is_last_job_for_agent:
                        is_agent_last_job = True # 是最终考核点
                        seen_agents_on_last_machine.add(agent_id)
                
                # 调用新的辅助函数寻找最佳开始时间
                best_start_time = self._find_best_start_time(est, lst, proc_time, is_last_machine, is_agent_last_job)
                

                
                # 2c. 根据找到的最佳开始时间, 更新该工序的最终完成时间
                final_ct_matrix[job_id, i] = best_start_time + proc_time

        # 3. 后处理: 将计算出的最终调度结果直接存入解中
        child_solution.final_schedule = final_ct_matrix
        child_solution.put_off = np.zeros_like(child_solution.put_off) # put_off不再使用, 但保持结构完整



        self.decoder.decode(child_solution)
        return child_solution

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

                # 使用新的工具函数确保不跨段
                start_time = self._first_valid_start(est, proc_time)
                
                completion_times[job_id, i] = start_time + proc_time
                
        return completion_times