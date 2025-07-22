from config import ProblemDefinition, Solution
from data_loader import load_problem_from_file
from main_algorithm import EvolutionaryAlgorithm
from results_handler import save_and_plot_results
import os

def main():
    DATA_FILE_PATH = "dataset\\data_A3_J7_M3_1.txt"
    
    # 将所有参数打包到一个字典中, 以便保存
    config = {
        'POP_SIZE': 50,
        'MAX_GENERATIONS': 20, # 减少代数以便快速测试
        'BFO_PARAMS': {
            'Mmax': 10,
            'C_initial': 0.1,
            'C_final': 0.01,
            'put_off_mutation_prob': 0.5,
            'put_off_mutation_strength': 5
        },
        'INIT_PARAMS': {
            'h1_count': 1,
            'h2_count': 1,
            'mutation_swaps': 30
        },
        'PROB_PARAMS': {
            'prob_crossover': 0.5,
            'prob_chemotaxis': 0.2,
            'prob_prefer_agent': 0.1,
            'prob_right_shift': 0.2, # 增加right_shift的探索概率
            'prob_migration': 0.1,
            'polishing_phase_gens': 5, # 精修5代
            'destroy_rebuild_alpha': 0.4
        }
    }
    
    try:
        problem_def = load_problem_from_file(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到, 请检查路径: {DATA_FILE_PATH}")
        return
    
    algorithm = EvolutionaryAlgorithm(
        problem_def=problem_def,
        pop_size=config['POP_SIZE'],
        max_generations=config['MAX_GENERATIONS'],
        bfo_params=config['BFO_PARAMS'],
        init_params=config['INIT_PARAMS'],
        prob_params=config['PROB_PARAMS']
    )
    
    final_population = algorithm.run()
    
    # 从数据文件路径中提取问题名称作为子文件夹名
    problem_name = os.path.splitext(os.path.basename(DATA_FILE_PATH))[0]
    output_folder = os.path.join("results", problem_name)
    
    save_and_plot_results(final_population, problem_def, output_folder, config)
    
if __name__ == "__main__":
    main()