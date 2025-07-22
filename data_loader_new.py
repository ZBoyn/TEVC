import numpy as np
from pro_def import ProblemDefinition
import ast
import os

def load_problem_g_format(file_path: str) -> ProblemDefinition:
    
    raw_data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                
                key, value_part = line.split('=', 1)
                key = key.strip()
                value_str = value_part.strip().rstrip(';')

                try:
                    value = ast.literal_eval(value_str)
                    raw_data[key] = value
                except (ValueError, SyntaxError):
                    print(f"无法解析键 '{key}' 的值: '{value_str}'")
                    raw_data[key] = value_str
    except FileNotFoundError:
        print(f"错误: 数据文件未找到，请检查路径: {file_path}")
        raise
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        raise

    try:
        num_agents = raw_data['g']
        num_jobs = raw_data['job']
        num_machines = raw_data['machine']

        p_matrix_raw = np.array(raw_data['P'], dtype=float)
        e_matrix_raw = np.array(raw_data['E'], dtype=float)
        processing_times = p_matrix_raw.transpose()
        power_consumption = e_matrix_raw.transpose()

        problem_def = ProblemDefinition(
            processing_times=processing_times,
            power_consumption=power_consumption,
            release_times=np.array(raw_data['R'], dtype=int),
            agent_job_indices=np.array(raw_data['ag'], dtype=int),
            agent_weights=np.ones(num_agents, dtype=float),   # 假定代理的权重均为1
            period_start_times=np.array(raw_data['U'], dtype=int),
            period_prices=np.array(raw_data['W'], dtype=float)
        )

        assert problem_def.num_agents == num_agents, "代理数量与文件中'g'不匹配"
        assert problem_def.num_jobs == num_jobs, "工件数量与文件中'job'不匹配"
        assert problem_def.num_machines == num_machines, "机器数量与文件中'machine'不匹配"

    except (KeyError, ValueError) as e:
        print(f"数据格式错误或键缺失: {e}")
        raise
    
    print(f"问题加载成功: {problem_def.num_jobs}个工件, {problem_def.num_machines}台机器, {problem_def.num_agents}个代理。")
    return problem_def

if __name__ == "__main__":
    test_file_path = os.path.join("data2", "3M7N-1.txt")
    if os.path.exists(test_file_path):
        try:
            problem = load_problem_g_format(test_file_path)
        except Exception as e:
            print(e)
    else:
        print(f"测试文件未找到: {test_file_path}") 