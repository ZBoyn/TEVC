import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.gd import GD
import pandas as pd
import os

def read_result(instance_name: str):
    bb_instance_name = instance_name.replace('-', 'J-')
    ea_instance_name = instance_name.replace('-', 'N-')

    bb_file_path = os.path.join('BBresults', f'{bb_instance_name}.txt')
    ea_file_path = os.path.join('results', ea_instance_name, 'pareto_front.xlsx')

    if not os.path.exists(bb_file_path):
        print(f"错误: B&B 结果文件未找到: {bb_file_path}")
        return
    if not os.path.exists(ea_file_path):
        print(f"错误: 进化算法结果文件未找到: {ea_file_path}")
        return
    
    try:
        tec_vals, tcta_vals = [], []
        with open(bb_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            next(f)  
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tec, tcta = map(int, line.split())
                    tec_vals.append(tec)
                    tcta_vals.append(tcta)
                except ValueError:
                    break
        bb_df = pd.DataFrame({'TEC': tec_vals, 'TCTA': tcta_vals})
        ea_df = pd.read_excel(ea_file_path)
        ea_df = ea_df[['TEC', 'TCTA']]

        bb_df = bb_df.sort_values('TEC').reset_index(drop=True)
        ea_df = ea_df.sort_values('TEC').reset_index(drop=True)
        
        bb_df = bb_df.drop_duplicates().reset_index(drop=True)
        ea_df = ea_df.drop_duplicates().reset_index(drop=True)

        bb_np = bb_df.to_numpy()
        ea_np = ea_df.to_numpy()

    except Exception as e:
        print(f"读取数据文件时发生错误: {e}")
        return
    
    return bb_np, ea_np

def calculate_IGD(bb_np, ea_np):
    """
    计算IGD (Inverted Generational Distance) 指标
    """
    ind = GD(bb_np)
    return ind(ea_np)

if __name__ == "__main__":
    instances = ["3M7-1", "3M7-2", "3M7-3", "3M7-4", "3M7-5",
                 "3M8-1", "3M8-2", "3M8-3", "3M8-4", "3M8-5",
                 "3M9-1", "3M9-2", "3M9-3", "3M9-4", "3M9-5",
                 "3M10-1", "3M10-2", "3M10-3", "3M10-4", "3M10-5"]

    for instance_name in instances:
        bb_np, ea_np = read_result(instance_name)
        print("GD", calculate_IGD(bb_np, ea_np))