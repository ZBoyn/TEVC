from config import ProblemDefinition, Solution
from heuInit import Initializer

class EvolutionaryAlgorithm:
    """
    我们将在这里构建完整的主算法。
    它将维护整个种群，并调用其他模块执行进化。
    """
    def __init__(self, problem_def: ProblemDefinition, pop_size: int):
        self.problem = problem_def
        self.pop_size = pop_size
        self.population = [] # 种群将在这里被维护

    def run(self):
        # 1. 初始化
        initializer = Initializer(self.problem, self.pop_size)
        self.population = initializer.initialize_population(h1_count=1, h2_count=1)
        
        # 2. TODO: 实现解码器，计算初始种群的目标值
        # decoder = Decoder(self.problem)
        # for sol in self.population:
        #     sol.objectives = decoder.calculate_objectives(sol)
            
        # 3. TODO: BFO主循环 + PreferAgent + RightShift 等
        
        print("\n--- 算法框架占位 ---")
        print(f"第一个个体的序列: {self.population[0].sequence}")
        print(f"该个体的put_off矩阵 (初始为0):\n{self.population[0].put_off}")