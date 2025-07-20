import numpy as np
from typing import List
from config import Solution

def non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
    """
    对整个种群执行快速非支配排序

    Args:
        population (List[Solution]): 需要排序的种群

    Returns:
        List[List[Solution]]: 一个包含所有Pareto前沿的列表
                              eg. [[front1_sol1, front1_sol2], [front2_sol1, ...]]
    """
    # 临时存储每个解的支配信息
    dominating_info = []
    for sol in population:
        # sp: 我支配的解的索引集合, np: 支配我的解的数量
        sp = set()
        np_count = 0
        for i, other_sol in enumerate(population):
            if sol is other_sol:
                continue
            
            # 判断支配关系
            if is_dominated(sol, other_sol):
                sp.add(i)
            elif is_dominated(other_sol, sol):
                np_count += 1
        
        dominating_info.append({"sp": sp, "np": np_count})
    
    # 开始分层
    fronts = []
    
    # 找到第一个前沿 (np == 0)
    current_front_indices = [i for i, info in enumerate(dominating_info) if info['np'] == 0]
    
    rank_counter = 0
    while current_front_indices:
        # 将当前前沿的解存入结果列表, 分配rank
        front_solutions = []
        for i in current_front_indices:
            population[i].rank = rank_counter
            front_solutions.append(population[i])
        fronts.append(front_solutions)
        
        # 准备寻找下一个前沿
        next_front_indices = []
        for i in current_front_indices:
            for dominated_index in dominating_info[i]['sp']:
                dominating_info[dominated_index]['np'] -= 1
                if dominating_info[dominated_index]['np'] == 0:
                    next_front_indices.append(dominated_index)
                    
        current_front_indices = next_front_indices
        rank_counter += 1
    
    return fronts

def crowding_distance_assignment(front: List[Solution]) -> None:
    """
    为一个Pareto前沿内的所有解计算拥挤度

    Args:
        front (List[Solution]): 单个前沿, 包含多个Solution对象
    """
    if not front:
        return
    
    num_solutions = len(front)
    num_objectives = len(front[0].objectives)
    
    # 初始化所有拥挤度为0
    for sol in front:
        sol.crowding_distance = 0.0
    
    for i in range(num_objectives):
        # 根据当前目标值对前沿进行排序
        front.sort(key=lambda x: x.objectives[i])
        
        # 边界点的拥挤度设置为无穷大
        front[0].crowding_distance = np.inf
        front[-1].crowding_distance = np.inf
        
        # 获取当前目标的最小值和最大值
        min_value = front[0].objectives[i]
        max_value = front[-1].objectives[i]
        
        if max_value - min_value == 0:
            continue
        
        # 计算中间点的拥挤度
        for j in range(1, num_solutions - 1):
            distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]
            normalized_distance = distance / (max_value - min_value)
            front[j].crowding_distance += normalized_distance

def is_dominated(sol_a: Solution, sol_b: Solution) -> bool:
    """
    判断解A是否支配解B (最小化问题)
    A支配B当且仅当A在所有目标上都不比B差, 并且至少在一个目标上严格比B好

    Args:
        sol_a (Solution): 解 A
        sol_b (Solution): 解 B

    Returns:
        bool: 如果 A 支配 B 返回 True, 否则返回 False
    """
    # 支配关系: A 在所有目标上都不差于 B 且至少在一个目标上优于 B
    # all(A <= B)
    not_worse_in_all = np.all(sol_a.objectives <= sol_b.objectives)
    # any(A < B)
    better_in_at_least_one = np.any(sol_a.objectives < sol_b.objectives)
    
    return not_worse_in_all and better_in_at_least_one