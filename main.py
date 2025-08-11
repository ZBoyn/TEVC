from pro_def import ProblemDefinition, Solution
# from data_loader_new import load_problem_g_format
from data_loader import load_problem_from_file
from main_algorithm import EvolutionaryAlgorithm
from results_handler import save_and_plot_results
from parameters import CONFIG, DATA_FILE_PATH
import os
import time

def main():
    
    data_file_path = DATA_FILE_PATH
    
    try:
        problem_def = load_problem_from_file(data_file_path)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到, 请检查路径: {data_file_path}")
        return
    
    problem_name = os.path.splitext(os.path.basename(data_file_path))[0]
    output_folder = os.path.join("Individual_results", problem_name)
    CONFIG['PLOT_PARAMS']['output_folder'] = output_folder

    algorithm = EvolutionaryAlgorithm(
        problem_def=problem_def,
        config=CONFIG
    )
    
    start_time = time.time()
    final_population = algorithm.run()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"\n算法总执行时间: {elapsed_time:.2f} 秒")
    
    CONFIG['elapsed_time_seconds'] = elapsed_time

    save_and_plot_results(final_population, problem_def, output_folder, CONFIG)
    
if __name__ == "__main__":
    main()