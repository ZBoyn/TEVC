import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from config import Solution
import os

def save_and_plot_results(pareto_front: List[Solution], output_folder: str = "results"):
    """
    将Pareto前沿的解保存到CSV文件, 并绘制散点图

    Args:
        pareto_front (List[Solution]): 包含最优解的列表
        output_folder (str): 保存结果的文件夹路径
    """
    if not pareto_front:
        print("警告: Pareto前沿为空, 无法保存或绘图")
        return

    # 确保结果文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    results_data = []
    for sol in pareto_front:
        results_data.append({
            'TEC': sol.objectives[0],
            'TCTA': sol.objectives[1],
            'sequence': ','.join(map(str, sol.sequence)),
            'put_off': str(sol.put_off[sol.put_off > 0]) # 只记录非零的put_off值
        })
    
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_folder, "pareto_front.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")

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