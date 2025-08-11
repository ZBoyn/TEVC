from typing import List, Tuple, Optional, Deque
from bb_node import SearchNode
from bb_tools import BoundCalculator, PruningRules
from pro_def import ProblemDefinition
from collections import deque
import numpy as np
import os
from data_loader_new import load_problem_g_format
from data_loader import load_problem_from_file

class HeuristicBBSolver:
    def __init__(self, problem: ProblemDefinition):
        self.problem = problem
        self.bound_calculator = BoundCalculator()
        self.pruning_rules = PruningRules()

    def find_heuristic_solution(self, objective_type: str, verbose: bool = True) -> Optional[Tuple[List[int], np.ndarray]]:
        # 使用双端队列作为回溯栈
        backtrack_stack: Deque[Tuple[SearchNode, List[SearchNode]]] = deque()
        current_node = SearchNode(unscheduled_jobs=set(range(self.problem.num_jobs)))
        
        iteration_count = 0
        # 跟踪连续相同优先级的次数
        consecutive_same_priority_count = 0
        last_priority = None
        while True:
            iteration_count += 1
            if verbose:
                print(f"\n--- Iteration {iteration_count}, Stack Size: {len(backtrack_stack)} ---")
                print(f"Current node: Seq={current_node.sequence}")

            if not current_node.unscheduled_jobs:
                print(f"\n{'='*20} SUCCESS {'='*20}")
                print(f"成功找到一个完整序列，目标: {objective_type}")
                print(f"总迭代次数: {iteration_count}")
                return self._reconstruct_solution(current_node)

            # 1. 生成所有有效的子节点
            children = []
            for job_to_schedule in sorted(list(current_node.unscheduled_jobs)):
                for is_put_off in [False, True]:
                    if self.pruning_rules.should_prune(current_node, job_to_schedule, is_put_off, self.problem, objective_type):
                        continue
                    child_node = self._create_child_node(current_node, job_to_schedule, is_put_off)
                    if child_node:
                        self._calculate_and_set_bounds(child_node, objective_type)
                        children.append(child_node)
            
            # 2. 选择最佳子节点深入，其他子节点存入回溯栈
            if children:
                children.sort() # 按priority升序排序
                best_child = children.pop(0) # 取出最好的
                
                # 检查连续相同优先级
                current_priority = best_child.priority
                if last_priority is not None and abs(current_priority - last_priority) < 1e-6:  # 使用小阈值比较浮点数
                    consecutive_same_priority_count += 1
                    if verbose:
                        print(f"  -> 连续相同优先级计数: {consecutive_same_priority_count}")
                else:
                    consecutive_same_priority_count = 1
                
                last_priority = current_priority
                
                # 如果连续3次相同优先级，停止探索
                if consecutive_same_priority_count >= 3:
                    if verbose:
                        print(f"  -> 检测到连续3次相同优先级 ({current_priority:.2f})，停止探索")
                        print(f"  -> 当前探索序列: {current_node.sequence}")
                    return self._reconstruct_solution(current_node)
                
                if verbose:
                    print(f"  -> Diving into best child: Seq={best_child.sequence}, Prio={best_child.priority:.2f}")
                
                # 将当前节点和剩余的子节点存入回溯栈
                if children:
                    backtrack_stack.append((current_node, children))
                    if verbose:
                        print(f"  -> Pushed {len(children)} other children to backtrack stack.")

                current_node = best_child
            
            # 3. 如果没有有效子节点，进行回溯
            else:
                if verbose:
                    print(f"  -> Dead end. Backtracking...")
                if not backtrack_stack:
                    print(f"\n{'='*20} FAILURE {'='*20}")
                    print(f"未能找到完整序列，目标: {objective_type}. Backtrack stack is empty.")
                    return None
                
                # 从栈中恢复上一个节点和它的待选子节点列表
                parent_node, remaining_children = backtrack_stack.pop()
                
                if verbose:
                    print(f"  -> Backtracked to node Seq={parent_node.sequence}. Trying next best child.")
                
                # 从剩余的子节点中选择下一个最好的
                best_child = remaining_children.pop(0)
                
                # 回溯时重置连续相同优先级计数，因为选择了不同的路径
                consecutive_same_priority_count = 0
                last_priority = None
                
                # 如果这个分支还有剩余的子节点，重新入栈
                if remaining_children:
                    backtrack_stack.append((parent_node, remaining_children))

                current_node = best_child

    def _create_child_node(self, parent: SearchNode, job: int, is_put_off: bool) -> Optional[SearchNode]:
        last_completion_time = parent.completion_times_m1[-1] if parent.completion_times_m1.size > 0 else 0
        start_time_m1 = max(last_completion_time, self.problem.release_times[job])
        if is_put_off:
            current_period_idx = np.searchsorted(self.problem.period_start_times, start_time_m1, side='right') - 1
            if current_period_idx + 1 < len(self.problem.period_start_times):
                start_time_m1 = max(start_time_m1, self.problem.period_start_times[current_period_idx + 1])
        completion_time_m1 = start_time_m1 + self.problem.processing_times[job, 0]
        if completion_time_m1 > self.problem.deadline: return None
        new_sequence = parent.sequence + (job,)
        new_put_off = parent.put_off_decisions + (int(is_put_off),)
        new_unscheduled = parent.unscheduled_jobs - {job}
        new_completion_times = np.append(parent.completion_times_m1, completion_time_m1)
        return SearchNode(sequence=new_sequence, put_off_decisions=new_put_off, unscheduled_jobs=new_unscheduled, completion_times_m1=new_completion_times)

    def _calculate_and_set_bounds(self, node: SearchNode, objective_type: str):
        node.lower_bound_tec = self.bound_calculator.calculate_tec_lower_bound(node, self.problem)
        node.lower_bound_tcta = self.bound_calculator.calculate_tcta_lower_bound(node, self.problem)
        node.priority = node.lower_bound_tec if objective_type == 'TEC' else node.lower_bound_tcta

    def _reconstruct_solution(self, leaf_node: SearchNode) -> Tuple[List[int], np.ndarray]:
        sequence = list(leaf_node.sequence)
        put_off_matrix = np.zeros((self.problem.num_jobs, self.problem.num_machines), dtype=int)
        put_off_m1 = np.array(leaf_node.put_off_decisions)
        original_indices_order = np.array(sequence)
        put_off_matrix[original_indices_order, 0] = put_off_m1
        return sequence, put_off_matrix

if __name__ == "__main__":
    # data_dir = "data2"
    # file_path = os.path.join(data_dir, "3M8N-1.txt")
    
    data_dir = "dataset"
    file_path = os.path.join(data_dir, "data_A3_J400_M4_1.txt")
    
    try:
        problem = load_problem_from_file(file_path)
        
        print("\n--- 开始启发式搜索 ---")
        solver = HeuristicBBSolver(problem)
        
        print("\n[目标: TCTA]")
        solution_tcta = solver.find_heuristic_solution('TCTA', verbose=True)
        if solution_tcta:
            seq, pom = solution_tcta
            print(f"\n最终找到的序列 (TCTA): {seq}")
            print(f"最终的推迟决策 (TCTA, 仅第一列有效):\n{pom}")

        # print("\n[目标: TEC]")
        # solution_tec = solver.find_heuristic_solution('TEC', verbose=True)
        # if solution_tec:
        #     seq, pom = solution_tec
        #     print(f"\n最终找到的序列 (TEC): {seq}")
        #     print(f"最终的推迟决策 (TEC, 仅第一列有效):\n{pom}")

    except Exception as e:
        print(f"\n主程序发生错误: {e}")
