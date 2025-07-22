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
        self.plot_params = self.config.get('PLOT_PARAMS') # 使用 .get 以防没有绘图参数
        
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
                # 精修阶段: 对当前种群中的每个个体应用强大的局部搜索算子
                polished_offspring = []
                alpha = self.prob_params.get('destroy_rebuild_alpha', 0.5)
                
                print(f"精修阶段: 对 {len(self.population)} 个个体应用 Destroy & Rebuild + Right Shift (alpha={alpha})...")
                for i, parent_sol in enumerate(self.population):
                    neh_optimized_sol = self.ls_toolkit.destroy_rebuild(parent_sol, alpha)
                    final_sol = self.ls_toolkit.right_shift(neh_optimized_sol)
                    polished_offspring.append(final_sol)

                # 用精修后的解更新外部存档
                self._update_archive(polished_offspring)

                # 合并父代与子代种群
                combined_population = self.population + polished_offspring
                
                # 去重
                deduplicated_population = remove_duplicates(combined_population)
                
                print(f"合并与去重: {len(combined_population)} -> {len(deduplicated_population)} 个独特解")

                # NSGA-II 精英选择
                fronts = non_dominated_sort(deduplicated_population)
                self.population = selection(fronts, self.pop_size)
                
                print(f"精英选择完成, 新种群大小: {len(self.population)}")
                print("-" * 50)
            else:
                offspring_population = self._generate_offspring(gen)

                # 步骤B: 评估子代
                for sol in offspring_population:
                    self.decoder.decode(sol)
                
                # 步骤B.1: 更新外部存档
                self._update_archive(offspring_population)

                # 步骤C: 合并父代与子代
                combined_population = self.population + offspring_population
                
                # 步骤C.1: 移除重复解
                unique_population = remove_duplicates(combined_population)

                # 步骤D: NSGA-II 环境选择
                fronts = non_dominated_sort(unique_population)
                self.population = selection(fronts, self.pop_size)

            # 定期绘制和保存帕累托前沿图
            if self.plot_params and self.plot_params.get('plot_frequency', 0) > 0 and (gen + 1) % self.plot_params['plot_frequency'] == 0:
                output_folder = self.plot_params.get('output_folder', 'results/temp')
                plot_intermediate_front(self.archive, gen + 1, output_folder)

            # 打印日志
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
        """通过调用工具箱中的算子来生成子代
           概率性地选择不同的算子
           注意是两阶段的 第一阶段进行繁殖和改进 第二阶段注入多样性
            
        Args:
            current_gen (int): 当前代数

        Returns:
            List[Solution]: 生成的子代种群
        """
        # 动态步长计算
        progress = current_gen / self.max_generations
        bfo_params = self.bfo_toolkit.params
        c_initial = bfo_params.get('C_initial', 0.1)
        c_final = bfo_params.get('C_final', 0.01)
        current_step_size = c_initial - (c_initial - c_final) * progress

        # 定义算子概率
        prob_crossover = self.prob_params.get('prob_crossover', 0.4)
        prob_chemotaxis = self.prob_params.get('prob_chemotaxis', 0.2)
        prob_prefer_agent = self.prob_params.get('prob_prefer_agent', 0.2)
        prob_right_shift = self.prob_params.get('prob_right_shift', 0.2)
        
        temp_offspring = []
        while len(temp_offspring) < self.pop_size:
            
            rand_num = np.random.rand()
            
            # 交叉操作 (生成两个子代)
            if rand_num < prob_crossover:
                p1 = tournament_selection(self.population)
                p2 = tournament_selection(self.population)
                child1, child2 = self.bfo_toolkit.reproduction_crossover(p1, p2)
                temp_offspring.extend([child1, child2])
            
            # 趋向操作
            elif rand_num < prob_crossover + prob_chemotaxis:
                p1 = tournament_selection(self.population)
                child = self.bfo_toolkit.chemotaxis(p1, current_step_size)
                temp_offspring.append(child)
            
            # 优势代理优化 (TCTA偏向)
            elif rand_num < prob_crossover + prob_chemotaxis + prob_prefer_agent:
                parent = tournament_selection(self.population)
                child = self.ls_toolkit.prefer_agent(parent)
                temp_offspring.append(child)

            # 右移优化 (TEC偏向)
            elif rand_num < prob_crossover + prob_chemotaxis + prob_prefer_agent + prob_right_shift:
                parent = tournament_selection(self.population)
                child = self.ls_toolkit.right_shift(parent)
                temp_offspring.append(child)
            
            else:
                p1 = tournament_selection(self.population)
                child = self.bfo_toolkit.chemotaxis(p1, current_step_size)
                temp_offspring.append(child)

        # 确保种群大小精确
        temp_offspring = temp_offspring[:self.pop_size]
        
        # 在生成后, 需要先对这个临时种群进行评估, 以便迁徙算子使用
        for sol in temp_offspring:
            self.decoder.decode(sol)

        # 多样性注入
        # 将整个临时子代种群送入种群级别的迁徙算子
        final_offspring = self.bfo_toolkit.migration(temp_offspring)
        
        return final_offspring