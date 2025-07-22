import numpy as np
from config import ProblemDefinition
import re
import os


def load_problem_from_file(file_path: str) -> ProblemDefinition:
    print(f"正在从文件加载问题: {file_path}")
    
    filename = os.path.basename(file_path)
    match = re.search(r'data_A(\d+)_J(\d+)_M(\d+)_', filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"文件名格式不正确，无法解析维度: {filename}")
    
    num_agents = int(match.group(1))   # A
    num_jobs = int(match.group(2))     # N (工件数)
    num_machines = int(match.group(3)) # M (机器数)

    print(f"从文件名解析得到: {num_agents}个代理, {num_jobs}个工件, {num_machines}台机器。")

    raw_data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                
                parts = line.split('=', 1)
                key_part = parts[0].strip()
                value_str = parts[1].strip()

                # 从键中提取一个简化的key
                simple_key_match = re.search(r'([A-Z]+)\[', key_part)
                if simple_key_match:
                    simple_key = simple_key_match.group(1)
                    raw_data[simple_key] = [val for val in value_str.split(',') if val]

    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请检查路径: {file_path}")
        raise
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        raise

    try:
        p_matrix_raw = np.array(raw_data['P'], dtype=float).reshape((num_machines, num_jobs))
        e_matrix_raw = np.array(raw_data['E'], dtype=float).reshape((num_machines, num_jobs))

        processing_times = p_matrix_raw.transpose()
        power_consumption = e_matrix_raw.transpose()
        
        problem_def = ProblemDefinition(
            processing_times=processing_times,
            power_consumption=power_consumption,
            release_times=np.array(raw_data['R'], dtype=int),
            agent_job_indices=np.array(raw_data['AN'], dtype=int),
            agent_weights=np.array(raw_data['AW'], dtype=float),
            period_start_times=np.array(raw_data['U'], dtype=int),
            period_prices=np.array(raw_data['W'], dtype=float)
        )
        
        assert problem_def.num_agents == num_agents, "代理数量与文件名不匹配"
        assert problem_def.num_jobs == num_jobs, "工件数量与文件名不匹配"
        assert problem_def.num_machines == num_machines, "机器数量与文件名不匹配"

    except (KeyError, ValueError) as e:
        print(f"数据格式错误: 文件内容与从文件名解析的维度不匹配。错误: {e}")
        raise

    print(f"问题加载成功: {problem_def.num_jobs}个工件, {problem_def.num_machines}台机器, {problem_def.num_agents}个代理。")
    return problem_def

if __name__ == "__main__":
    file_path = "dataset\data_A3_J400_M4_2.txt"
    try:
        problem = load_problem_from_file(file_path)
        print("问题定义加载成功:", problem)
    except Exception as e:
        print("加载问题定义失败:", e)