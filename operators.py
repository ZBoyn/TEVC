import numpy as np
from config import ProblemDefinition, Solution
from moeaTools import is_dominated
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
                        
            # 1c. 构造并评估候选解
            trail_sol = Solution(sequence=new_sequence, put_off=new_put_off)
            self.decoder.decode(trail_sol)
            
            # 2. 基于Pareto支配关系决定是否接受移动
            if is_dominated(trail_sol, current_sol):
                # 如果新解支配当前解, 则接受新解
                current_sol = trail_sol
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
        put_off1, put_off2 = parent1.put_off, parent2.put_off
        mask = np.random.rand(*put_off1.shape) < 0.5
        
        child1_put_off = np.where(mask, put_off1, put_off2)
        child2_put_off = np.where(mask, put_off2, put_off1)

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