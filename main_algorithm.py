from pro_def import ProblemDefinition, Solution
from heu_init import Initializer
import numpy as np
from decode import Decoder
from typing import List
from moea_tools import non_dominated_sort, remove_duplicates, selection, tournament_selection
from operators import BFO_Operators, LocalSearch_Operators
from results_handler import plot_intermediate_front
import os


class EvolutionaryAlgorithm:
    """
    算法主框架
    以NSGA-II为骨架 集成BFO和问题特有的局部搜索算子
    """
    def __init__(self, problem_def: ProblemDefinition, config: dict):
        self.problem = problem_def
        self.config = config
        
        # 从配置中解析参数
        self.pop_size = self.config['POP_SIZE']
        self.max_generations = self.config['MAX_GENERATIONS']
        
        self.init_params = self.config['INIT_PARAMS']
        self.prob_params = self.config['PROB_PARAMS']
        self.plot_params = self.config.get('PLOT_PARAMS')
        
        self.initializer = Initializer(self.problem, self.pop_size, self.init_params)
        self.decoder = Decoder(self.problem)

        # 实例化工具箱
        self.bfo_toolkit = BFO_Operators(self.problem, self.decoder, self.config['BFO_PARAMS'])
        self.ls_toolkit = LocalSearch_Operators(self.problem, self.decoder)

        self.population: List[Solution] = []
        self.archive: List[Solution] = []
        self.polishing_phase_start_gen: int = self.max_generations - self.prob_params.get('polishing_phase_gens', 5)   # 执行NEH+RightShift的精修阶段


    def run(self):
        """执行完整的多目标进化算法流程"""
        # 初始化
        # 生成初始种群 (sequence + 全0的put_off矩阵)
        self.population = self.initializer.initialize_population()

        # 评估初始种群
        for sol in self.population:
            self.decoder.decode(sol)

        # 初始化外部存档
        self._update_archive(self.population)

        # 主进化
        for gen in range(self.max_generations):
            print(f"\n第 {gen + 1}/{self.max_generations} 代进化")
            
            # 判断是否进入精修阶段
            if gen >= self.polishing_phase_start_gen:
                # 精修阶段: 概率性地对种群中的个体应用强大的局部搜索算子
                polished_offspring = []
                alpha = self.prob_params.get('destroy_rebuild_alpha', 0.5)
                prob_polish = self.prob_params.get('prob_polish', 0.4)
                
                print(f"精修阶段: 以 {prob_polish*100}% 的概率对个体应用 Destroy & Rebuild + Right Shift...")
                
                for parent_sol in self.population:
                    if np.random.rand() < prob_polish:
                        # 应用强力优化算子组合
                        # neh_optimized_sol = self.ls_toolkit.destroy_rebuild(parent_sol, alpha)
                        # neh_optimized_sol.generated_by = "Destroy&Rebuild"
                        final_sol = self.ls_toolkit.right_shift(parent_sol)
                        final_sol.generated_by = "Destroy&Rebuild+RightShift"
                        polished_offspring.append(final_sol)
                    else:
                        # 未被选中的直接进入下一代
                        polished_offspring.append(parent_sol.copy())

                # 用精修后的新解更新外部存档 (注意, polished_offspring中包含了未改变的个体)
                self._update_archive(polished_offspring)

                # 环境选择: 直接将精修/幸存的个体作为新种群
                # 因为这里的目标是提纯, 而不是扩大搜索, 所以不与父代合并
                self.population = remove_duplicates(polished_offspring)
                
                print(f"精英提纯完成, 新种群大小: {len(self.population)}")
                print("-" * 50)
            else:
                # 常规进化阶段
                offspring_population = self._generate_offspring(gen)
                
                # 2. 更新外部存档
                self._update_archive(offspring_population)
                
                # 3. 环境选择 (合并父子代, 去重, NSGA-II选择)
                combined_population = self.population + offspring_population
                unique_population = remove_duplicates(combined_population)
                fronts = non_dominated_sort(unique_population)
                self.population = selection(fronts, self.pop_size)

            if self.plot_params and self.plot_params.get('plot_frequency', 0) > 0 and (gen + 1) % self.plot_params['plot_frequency'] == 0:
                output_folder = self.plot_params.get('output_folder', 'results/temp')
                plot_intermediate_front(self.archive, gen + 1, output_folder)

            print(f"新种群选择完毕。外部存档中最优解数量: {len(self.archive)}")

        # 算法结束, 返回外部存档中的所有最优解
        return self.archive
    
    def _update_archive(self, new_solutions: List[Solution]):
        """
        使用新生成的解来更新外部存档.
        存档中只保留全局非支配解.
        同时进行基因型和表现型去重.
        """
        # 1. 合并当前存档和新解
        combined_archive = self.archive + new_solutions
        
        # 2. 基因型去重 (基于解的编码)
        genotype_unique_solutions = remove_duplicates(combined_archive)

        # 3. 对基因型唯一的解进行非支配排序
        fronts = non_dominated_sort(genotype_unique_solutions)
        
        if not fronts:
            return

        # 4. 提取Pareto前沿, 并进行表现型去重 (基于目标函数值)
        potential_archive = fronts[0]
        phenotype_unique_solutions = {}
        for sol in potential_archive:
            objectives_tuple = tuple(sol.objectives)
            if objectives_tuple not in phenotype_unique_solutions:
                phenotype_unique_solutions[objectives_tuple] = sol
        
        self.archive = list(phenotype_unique_solutions.values())

    def _generate_offspring(self, current_gen: int) -> List[Solution]:
        """
        通过标准的进化流程(交配池、交叉、变异)生成并评估子代种群
        """
        # 动态步长
        progress = current_gen / self.max_generations if self.max_generations > 0 else 0
        bfo_params = self.bfo_toolkit.params
        c_initial = bfo_params.get('C_initial', 0.1)
        c_final = bfo_params.get('C_final', 0.01)
        current_step_size = c_initial - (c_initial - c_final) * progress

        # 1. 根据进化阶段动态调整算子池
        exploration_ratio = self.prob_params.get('exploration_phase_ratio', 0.3)  # 前 30% 代主要探索

        # 基础算子
        mutation_operators = [
            lambda sol: self.bfo_toolkit.chemotaxis(sol, current_step_size), # 趋化算子
            self.ls_toolkit.prefer_agent
        ]

        # 中后期再加入较强的 right_shift
        if progress >= exploration_ratio:
            mutation_operators.append(self.ls_toolkit.right_shift) # 右移算子

        # 算子概率
        prob_crossover = self.prob_params.get('prob_crossover', 0.9)
        prob_mutation = self.prob_params.get('prob_mutation', 0.2)

        # 2. 选择: 创建交配池
        mating_pool = [tournament_selection(self.population) for _ in range(self.pop_size)]

        # 3. 繁殖: 交叉与变异
        offspring_from_reproduction = []
        i = 0
        while len(offspring_from_reproduction) < self.pop_size:
            p1 = mating_pool[i]
            # 确保有p2, 即使种群大小为奇数
            p2 = mating_pool[i + 1] if i + 1 < self.pop_size else mating_pool[0]
            i = (i + 2) % self.pop_size  # 循环使用交配池

            # 交叉
            if np.random.rand() < prob_crossover:
                c1, c2 = self.bfo_toolkit.reproduction_crossover(p1, p2)
                c1.generated_by = "crossover"
                c2.generated_by = "crossover"
            else:
                c1, c2 = p1.copy(), p2.copy()

            # 变异
            for child in [c1, c2]:
                if np.random.rand() < prob_mutation:
                    operator = np.random.choice(mutation_operators)
                    mutated_child = operator(child)
                    # 从函数引用中获取可读的算子名称
                    operator_name = getattr(operator, '__name__', 'unknown_operator')
                    if hasattr(operator, '__self__'):
                        operator_name = f"{operator.__self__.__class__.__name__}.{operator_name}"
                    mutated_child.generated_by = f"mutation:{operator_name}"
                    offspring_from_reproduction.append(mutated_child)
                else:
                    offspring_from_reproduction.append(child)

        # 保证种群大小精确
        offspring_population = offspring_from_reproduction[:self.pop_size]

        # 4. 周期性迁徙操作 (大步刷新多样性)
        migration_freq = self.prob_params.get('migration_freq', 10)
        if migration_freq > 0 and (current_gen + 1) % migration_freq == 0:
            print(f"触发迁徙算子: 第 {current_gen + 1} 代, 对子代进行重置以提升多样性…")
            migrated_offspring = self.bfo_toolkit.migration(offspring_population)
            for sol in migrated_offspring:
                sol.generated_by = "migration"
            offspring_population = migrated_offspring

        # 5. 评估
        for sol in offspring_population:
            if sol.objectives is None:
                self.decoder.decode(sol)

        return offspring_population