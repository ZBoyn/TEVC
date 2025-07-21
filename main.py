from config import ProblemDefinition, Solution
from data_loader import load_problem_from_file
from main_algorithm import EvolutionaryAlgorithm
from results_handler import save_and_plot_results

def main():
    DATA_FILE_PATH = "dataset\data_A3_J400_M4_1.txt"
    
    # 算法参数
    POP_SIZE = 50
    MAX_GENERATIONS = 100
    
    # BFO算子参数
    BFO_PARAMS = {
        'Mmax': 10,
        'C_initial': 0.1,
        'C_final': 0.01,
        'put_off_mutation_prob': 0.1,
        'put_off_mutation_strength': 2
    }
    
    # 初始化参数
    INIT_PARAMS = {
        'h1_count': 1,  # 使用Heuristic 1生成5个解
        'h2_count': 1,  # 使用Heuristic 2生成5个解
        'mutation_swaps': 30  # 随机生成30次交换的解
    }
    
    try:
        problem_def = load_problem_from_file(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到, 请检查路径: {DATA_FILE_PATH}")
        return
    
    algorithm = EvolutionaryAlgorithm(
        problem_def=problem_def,
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        bfo_params=BFO_PARAMS,
        init_params=INIT_PARAMS
    )
    
    final_population = algorithm.run()
    pareto_front_solutions = final_population
    save_and_plot_results(pareto_front_solutions, output_folder="results")
    
if __name__ == "__main__":
    main()