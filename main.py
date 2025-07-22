from config import ProblemDefinition, Solution
from data_loader import load_problem_from_file
from data_loader_new import load_problem_g_format
from main_algorithm import EvolutionaryAlgorithm
from results_handler import save_and_plot_results
import os
import time

def main():
    DATA_FILE_PATH = "data2\\3M10N-14.txt"
    
    # 将所有参数打包到一个字典中, 以便保存
    config = {
        'POP_SIZE': 50,
        'MAX_GENERATIONS': 500,
        'PLOT_PARAMS': {
            'plot_frequency': 10  # 每 10 代绘制一次前沿图
        },
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
            'prob_crossover': 0.3,
            'prob_chemotaxis': 0.3,
            'prob_prefer_agent': 0.2,
            'prob_right_shift': 0.2, # 增加right_shift的探索概率
            'prob_migration': 0.1,
            'polishing_phase_gens': 30,
            'destroy_rebuild_alpha': 0.5
        }
    }
    
    try:
        problem_def = load_problem_g_format(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到, 请检查路径: {DATA_FILE_PATH}")
        return
    
    # 从数据文件路径中提取问题名称作为子文件夹名
    problem_name = os.path.splitext(os.path.basename(DATA_FILE_PATH))[0]
    output_folder = os.path.join("results", problem_name)
    
    # 将输出文件夹也添加到plot_params中, 方便传递
    config['PLOT_PARAMS']['output_folder'] = output_folder

    algorithm = EvolutionaryAlgorithm(
        problem_def=problem_def,
        pop_size=config['POP_SIZE'],
        max_generations=config['MAX_GENERATIONS'],
        bfo_params=config['BFO_PARAMS'],
        init_params=config['INIT_PARAMS'],
        prob_params=config['PROB_PARAMS'],
        plot_params=config['PLOT_PARAMS']
    )
    
    start_time = time.time()
    final_population = algorithm.run()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\n算法总执行时间: {elapsed_time:.2f} 秒")
    
    # 将执行时间保存到config中
    config['elapsed_time_seconds'] = elapsed_time

    save_and_plot_results(final_population, problem_def, output_folder, config)
    
if __name__ == "__main__":
    main()