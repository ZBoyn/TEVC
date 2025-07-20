import numpy as np
from config import ProblemDefinition, Solution
from moeaTools import is_dominated

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