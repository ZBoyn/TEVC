import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from config import Solution, ProblemDefinition
import os
import numpy as np
import json

def save_and_plot_results(pareto_front: List[Solution], problem_def: ProblemDefinition, output_folder: str, config: dict):
    """
    将Pareto前沿的解保存到CSV文件, 绘制散点图, 并保存当次运行的参数配置

    Args:
        pareto_front (List[Solution]): 包含最优解的列表
        problem_def (ProblemDefinition): 问题定义, 用于计算时段
        output_folder (str): 保存结果的文件夹路径 (例如 "results/data_A3_J7_M3_1")
        config (dict): 包含当次运行所有参数的字典
    """
    if not pareto_front:
        print("警告: Pareto前沿为空, 无法保存或绘图")
        return

    # 确保结果文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    results_data = []
    for sol in pareto_front:
        period_schedule_str = "N/A"
        if sol.completion_times is not None:
            # 根据完成时间计算每个工序的开始时间
            start_times = sol.completion_times - problem_def.processing_times
            
            # 找到每个开始时间对应的时段索引
            period_indices = np.searchsorted(problem_def.period_start_times, start_times, side='right') - 1
            
            # 将二维数组格式化为字符串以便存储
            period_schedule_str = ';'.join(','.join(map(str, row)) for row in period_indices)

        results_data.append({
            'TEC': sol.objectives[0],
            'TCTA': sol.objectives[1],
            'sequence': ','.join(map(str, sol.sequence)),
            'period_schedule': period_schedule_str
        })
    
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_folder, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")

    # 保存参数配置
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"参数配置已保存到: {config_path}")

    tec_values = [sol.objectives[0] for sol in pareto_front]
    tcta_values = [sol.objectives[1] for sol in pareto_front]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tec_values, tcta_values, c='blue', marker='o', label='Pareto Optimal Solutions')
    plt.title('Pareto Front (TEC vs. TCTA)', fontsize=16)
    plt.xlabel('Total Energy Consumption (TEC)', fontsize=12)
    plt.ylabel('Total Weighted Agent Completion Time (TCTA)', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(output_folder, "pareto_front.png")
    plt.savefig(plot_path)
    print(f"Pareto前沿图像已保存到: {plot_path}")
    plt.close()

def plot_intermediate_front(archive: List[Solution], current_gen: int, output_folder: str):
    """
    在进化过程中绘制并保存当前的帕累托前沿图

    Args:
        archive (List[Solution]): 当前的外部存档
        current_gen (int): 当前代数
        output_folder (str): 保存图像的文件夹路径
    """
    if not archive:
        print(f"警告: 第 {current_gen} 代的外部存档为空, 跳过绘图.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    objectives = np.array([sol.objectives for sol in archive])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(objectives[:, 0], objectives[:, 1], c='b', marker='o', label=f'Generation {current_gen}')
    
    plt.title(f'Pareto Front at Generation {current_gen}')
    plt.xlabel("Total Energy Consumption (TEC)")
    plt.ylabel("Total Weighted Agent Completion Time (TCTA)")
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(output_folder, f'pareto_front_gen_{current_gen}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"已保存第 {current_gen} 代的前沿图像到: {plot_filename}")