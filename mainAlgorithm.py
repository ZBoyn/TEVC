from config import ProblemDefinition, Solution
from heuInit import Initializer
import numpy as np
from decode import Decoder
from typing import List
from moeaTools import non_dominated_sort, is_dominated, crowding_distance_assignment
from operators import BFO_Operators

class EvolutionaryAlgorithm:
    """
    算法主框架
    以NSGA-II为骨架 集成BFO和问题特有的局部搜索算子
    """
    def __init__(self, problem_def: ProblemDefinition, pop_size: int, max_generations: int, bfo_params: dict):
        self.problem = problem_def
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        self.initializer = Initializer(self.problem, self.pop_size)
        self.decoder = Decoder(self.problem)

        # 实例化工具箱
        self.bfo_toolkit = BFO_Operators(self.problem, self.decoder, bfo_params)

        self.population: List[Solution] = []


    def run(self):
        """执行完整的多目标进化算法流程"""
        # 初始化
        # 生成初始种群 (sequence + 全0的put_off矩阵)
        self.population = self.initializer.initialize_population(h1_count=1, h2_count=1, mutation_swaps=30)
        
        # 评估初始种群
        for sol in self.population:
            self.decoder.decode(sol)

        # 主进化
        for gen in range(self.max_generations):
            print(f"\n第 {gen + 1}/{self.max_generations} 代进化")
            
            # 步骤A: 生成子代种群
            offspring_population = self._generate_offspring(gen)

            # 步骤B: 评估子代
            for sol in offspring_population:
                self.decoder.decode(sol)

            # 步骤C: 合并父代与子代
            combined_population = self.population + offspring_population

            # 步骤D: NSGA-II 环境选择
            fronts = non_dominated_sort(combined_population)
            self.population = self._selection(fronts)

            # 打印日志
            print(f"新种群选择完毕。最优前沿解数量: {len(fronts[0])}")

        # 算法结束
        # final_front = non_dominated_sort(self.population)[0]
        # print(f"\n算法结束。最终Pareto前沿包含 {len(final_front)} 个解。")
        # return final_front
        return self.population # 暂时返回最终种群
    
    def _generate_offspring(self, current_gen: int) -> List[Solution]:
        """通过调用工具箱中的算子来生成子代

        Args:
            current_gen (int): 当前代数

        Returns:
            List[Solution]: 生成的子代种群
        """
        # 动态步长计算
        progress = current_gen / self.max_generations
        c_initial = self.bfo_toolkit.params.get('C_initial', 0.1)
        c_final = self.bfo_toolkit.params.get('C_final', 0.01)
        current_step_size = c_initial - (c_initial - c_final) * progress

        offspring = []
        # 示例：一半个体执行趋向，一半执行其他操作
        for i in range(self.pop_size):
            parent = self.population[i]
            # 这里可以设计更复杂的算子选择策略
            # 现在我们只实现趋向操作
            new_sol = self.bfo_toolkit.chemotaxis(parent, current_step_size)
            offspring.append(new_sol)
        return offspring

    def _selection(self, fronts: List[List[Solution]]) -> List[Solution]:
        """执行NSGA-II的选择操作, 填充下一代种群

        Args:
            fronts (List[List[Solution]]): 包含所有Pareto前沿的列表

        Returns:
            List[Solution]: 选择后的下一代种群
        """
        next_population = []
        for front in fronts:
            if len(next_population) + len(front) <= self.pop_size:
                next_population.extend(front)
            else:
                # 如果当前前沿无法完全放入，则计算拥挤度并选择
                crowding_distance_assignment(front)
                # 按拥挤度降序排序
                front.sort(key=lambda sol: sol.crowding_distance, reverse=True)
                remaining_space = self.pop_size - len(next_population)
                next_population.extend(front[:remaining_space])
                break
        return next_population