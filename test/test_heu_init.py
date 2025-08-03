import os
import numpy as np
from data_loader import load_problem_from_file
from heu_init import Initializer

def test_heu_init_with_bb_heu():
    """测试heu_init与bb_heu的集成"""
    
    # 加载问题
    data_dir = "dataset"
    file_path = os.path.join(data_dir, "data_A3_J400_M4_1.txt")
    
    try:
        problem = load_problem_from_file(file_path)
        print(f"成功加载问题: {file_path}")
        print(f"工件数: {problem.num_jobs}, 机器数: {problem.num_machines}")
        
        # 创建初始化器
        init_params = {
            'h1_count': 2,
            'h2_count': 2,
            'mutation_swaps': 10,
            'random_init_ratio': 0.3
        }
        
        initializer = Initializer(problem, pop_size=20, init_params=init_params)
        
        # 获取来自bb_heu的不完整解
        print("\n=== 获取来自bb_heu的不完整解 ===")
        partial_solutions = initializer.get_partial_solutions_from_bb_heu()
        print(f"获得 {len(partial_solutions)} 个不完整解")
        
        # 使用不完整解初始化种群
        print("\n=== 使用不完整解初始化种群 ===")
        population = initializer.initialize_population(partial_solutions)
        
        print(f"\n种群初始化完成，共 {len(population)} 个个体")
        
        # 显示每个个体的信息
        for i, solution in enumerate(population):
            print(f"个体 {i+1}: 序列长度={len(solution.sequence)}, 生成方式={solution.generated_by}")
        
        # 验证所有序列都是完整的
        all_complete = all(len(sol.sequence) == problem.num_jobs for sol in population)
        print(f"\n所有序列都是完整的: {all_complete}")
        
        # 验证所有序列都是有效的（包含所有工件且无重复）
        all_valid = True
        for i, solution in enumerate(population):
            seq = solution.sequence
            if len(set(seq)) != len(seq) or len(seq) != problem.num_jobs:
                print(f"个体 {i+1} 序列无效: {seq}")
                all_valid = False
            elif set(seq) != set(range(problem.num_jobs)):
                print(f"个体 {i+1} 序列不包含所有工件: {seq}")
                all_valid = False
        
        print(f"所有序列都是有效的: {all_valid}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_heu_init_with_bb_heu() 