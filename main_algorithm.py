from config import ProblemDefinition, Solution
from heu_init import Initializer
import numpy as np
from decode import Decoder
from typing import List
from moea_tools import non_dominated_sort, is_dominated, crowding_distance_assignment
from operators import BFO_Operators, LocalSearch_Operators

class EvolutionaryAlgorithm:
    """
    算法主框架
    以NSGA-II为骨架 集成BFO和问题特有的局部搜索算子
    """
    def __init__(self, problem_def: ProblemDefinition, pop_size: int, max_generations: int, bfo_params: dict, init_params: dict, prob_params: dict):
        self.problem = problem_def
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        self.init_params = init_params
        self.prob_params = prob_params
        self.initializer = Initializer(self.problem, self.pop_size)
        self.decoder = Decoder(self.problem)

        # 实例化工具箱
        self.bfo_toolkit = BFO_Operators(self.problem, self.decoder, bfo_params)
        self.ls_toolkit = LocalSearch_Operators(self.problem, self.decoder)

        self.population: List[Solution] = []
        self.archive: List[Solution] = []
        self.polishing_phase_start_gen: int = max_generations - self.prob_params.get('polishing_phase_gens', 5)   # 执行NEH+RightShift的精修阶段


    def run(self):
        """执行完整的多目标进化算法流程"""
        # 初始化
        # 生成初始种群 (sequence + 全0的put_off矩阵)
        h1_count = self.init_params.get('h1_count', 1)
        h2_count = self.init_params.get('h2_count', 1)
        mutation_swaps = self.init_params.get('mutation_swaps', 30)
        self.population = self.initializer.initialize_population(h1_count=h1_count, h2_count=h2_count, mutation_swaps=mutation_swaps)

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
                offspring_population = self._polish_population()
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
            unique_population = self._remove_duplicates(combined_population)

            # 步骤D: NSGA-II 环境选择
            fronts = non_dominated_sort(unique_population)
            self.population = self._selection(fronts)

            # 打印日志
            print(f"新种群选择完毕。外部存档中最优解数量: {len(self.archive)}")

        # 算法结束, 返回外部存档中的所有最优解
        return self.archive
    
    def _update_archive(self, new_solutions: List[Solution]):
        """
        使用新生成的解来更新外部存档.
        存档中只保留全局非支配解.
        """
        combined_archive = self.archive + new_solutions
        unique_candidates = self._remove_duplicates(combined_archive)
        fronts = non_dominated_sort(unique_candidates)
        if fronts:
            self.archive = fronts[0]

    def _remove_duplicates(self, population: List[Solution]) -> List[Solution]:
        """
        移除种群中的重复解.
        通过将 sequence 和 put_off 矩阵转换为可哈希的元组来实现.
        """
        unique_solutions = []
        seen_signatures = set()

        for sol in population:
            sequence_tuple = tuple(sol.sequence)
            put_off_tuple = tuple(map(tuple, sol.put_off))
            signature = (sequence_tuple, put_off_tuple)
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_solutions.append(sol)
                
        return unique_solutions

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
                p1 = self._tournament_selection()
                p2 = self._tournament_selection()
                child1, child2 = self.bfo_toolkit.reproduction_crossover(p1, p2)
                temp_offspring.extend([child1, child2])
            
            # 趋向操作
            elif rand_num < prob_crossover + prob_chemotaxis:
                p1 = self._tournament_selection()
                child = self.bfo_toolkit.chemotaxis(p1, current_step_size)
                temp_offspring.append(child)
            
            # 优势代理优化 (TCTA偏向)
            elif rand_num < prob_crossover + prob_chemotaxis + prob_prefer_agent:
                parent = self._tournament_selection()
                child = self.ls_toolkit.prefer_agent(parent)
                temp_offspring.append(child)

            # 右移优化 (TEC偏向)
            elif rand_num < prob_crossover + prob_chemotaxis + prob_prefer_agent + prob_right_shift:
                parent = self._tournament_selection()
                child = self.ls_toolkit.right_shift(parent)
                temp_offspring.append(child)
            
            else:
                p1 = self._tournament_selection()
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

    def _polish_population(self) -> List[Solution]:
        """
        精修阶段: 对当前种群中的每个个体应用强大的局部搜索算子.
        """
        polished_offspring = []
        alpha = self.prob_params.get('destroy_rebuild_alpha', 0.5) # 和日志中保持一致
        
        # 对种群中的每一个解应用destroy_rebuild
        print(f"对 {len(self.population)} 个个体应用 Destroy & Rebuild + Right Shift (alpha={alpha})...")
        for i, parent_sol in enumerate(self.population):
            original_objectives = parent_sol.objectives.copy()

            # 1. 先用 NEH 优化序列
            neh_optimized_sol = self.ls_toolkit.destroy_rebuild(parent_sol, alpha)
            
            # 2. 再用 right_shift 优化 put_off
            # 这一步确保了即使是精修阶段, 两个目标也都在被优化
            final_sol = self.ls_toolkit.right_shift(neh_optimized_sol)
            
            # 【调试关键】right_shift 内部已经解码, 直接获取最终的目标值
            new_objectives = final_sol.objectives

            # --- 打印详细的调试信息 ---
            print(f"  [个体 {i+1:02d}/{len(self.population)}] ", end="")
            print(f"原目标: [{original_objectives[0]:.2f}, {original_objectives[1]:.2f}] -> ", end="")
            print(f"新目标: [{new_objectives[0]:.2f}, {new_objectives[1]:.2f}] | ", end="")

            # is_dominated(A, B) 表示 B是否被A支配
            if is_dominated(final_sol, parent_sol):
                print("✅ 提升")
            elif is_dominated(parent_sol, final_sol):
                print("❌ 变差")
            elif np.allclose(original_objectives, new_objectives):
                print("⚪️ 无变化")
            else:
                print("🔵 非支配")
            # --- 调试信息结束 ---

            polished_offspring.append(final_sol)
            
        return polished_offspring
            

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
    
    def _tournament_selection(self) -> Solution:
        """通过二元锦标赛选择法, 从当前种群中选择一个个体

        Returns:
            Solution: 选择的个体
        """
        p1_idx, p2_idx = np.random.choice(self.pop_size, size=2, replace=False)
        parent1 = self.population[p1_idx]
        parent2 = self.population[p2_idx]
        
        # 根据Pareto等级和拥挤度决定优胜者
        if parent1.rank < parent2.rank:
            return parent1
        elif parent1.rank > parent2.rank:
            return parent2
        else:
            # 如果等级相同，比较拥挤度
            if parent1.crowding_distance > parent2.crowding_distance:
                return parent1
            else:
                return parent2