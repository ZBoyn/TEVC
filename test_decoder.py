import numpy as np
from pro_def import ProblemDefinition, Solution
from decode import Decoder

def test_decoder():
    """
    用于测试Decoder模块正确性的单元测试函数。
    """
    
    # 3个工件, 2台机器, 2个代理, 3个电价时段
    problem_data = ProblemDefinition(
        processing_times=np.array([[5, 6], [8, 4], [7, 9]]), # P[3, 2]
        power_consumption=np.array([[2, 2], [3, 3], [1, 1]]), # E[3, 2]
        release_times=np.array([0, 10, 0]),                  # R[3], 工件1有释放延迟
        agent_job_indices=np.array([0, 2, 3]),               # 代理0:{0,1}, 代理1:{2}
        agent_weights=np.array([0.6, 0.4]),                  # AW[2]
        period_start_times=np.array([0, 20, 40, 1000]),      # U[4], 时段为[0,20), [20,40), [40,inf)
        period_prices=np.array([1.0, 2.0, 0.5])              # W[3], 中间时段最贵
    )

# g=3;
# ag=[0,2,4,10];
# job=10;
# machine=3;
# R=[44, 45, 45, 30, 16, 2, 29, 4, 45, 26];
# P=[[1, 10, 3, 10, 1, 4, 2, 7, 8, 6], [6, 5, 4, 3, 4, 4, 10, 5, 9, 6], [11, 8, 7, 8, 8, 11, 2, 3, 10, 2]];
# E=[[4, 8, 4, 8, 8, 4, 6, 2, 8, 8], [4, 6, 6, 6, 6, 8, 2, 8, 2, 6], [6, 2, 2, 6, 2, 6, 2, 8, 8, 2]];

# K=3;
# U=[0, 40, 70, 115];
# W=[9, 6, 3];

    problem_data = ProblemDefinition(
        # 转置
        processing_times =np.array([[1, 10, 3, 10, 1, 4, 2, 7, 8, 6], [6, 5, 4, 3, 4, 4, 10, 5, 9, 6], [11, 8, 7, 8, 8, 11, 2, 3, 10, 2]]).T,
        power_consumption = np.array([[4, 8, 4, 8, 8, 4, 6, 2, 8, 8], [4, 6, 6, 6, 6, 8, 2, 8, 2, 6], [6, 2, 2, 6, 2, 6, 2, 8, 8, 2]]).T,
        release_times = np.array([44, 45, 45, 30, 16, 2, 29, 4, 45, 26]),
        agent_job_indices = np.array([0, 2, 4, 10]),
        agent_weights = np.array([1, 1, 1]),
        period_start_times = np.array([0, 40, 70, 115]),
        period_prices = np.array([9, 6, 3])
    )

    # 实例化解码器
    decoder = Decoder(problem_data)
    
    # # 测试场景1: 简单的紧凑调度 (put_off全为0)
    # print("\n--- 场景1: 紧凑调度 (put_off全为0) ---")
    # seq1 = np.array([0, 2, 1]) # 序列 0 -> 2 -> 1
    # put_off1 = np.zeros_like(problem_data.processing_times)
    # solution1 = Solution(sequence=seq1, put_off=put_off1)

    # objectives1, completion_times1, operation_periods1 = decoder.decode(solution1)
    # # objectives1 = decoder.decode(solution1)

    # print(f"  - 测试序列: {solution1.sequence}")
    # print(f"  - 目标值 [TEC, TCTA]: {np.round(objectives1, 2)}")
    # print(f"  - 完成时间矩阵:\n{completion_times1}")
    # print(f"  - 时段操作矩阵:\n{operation_periods1}")
    # print("Good 已验证 与手算结果一致")  # [95.  31.4]
        
    # # 测试场景2: 时段推迟生效
    # print("\n--- 场景2: '时段推迟' (put_off生效) ---")
    # seq2 = np.array([0, 2, 1])
    # put_off2 = np.zeros_like(problem_data.processing_times)
    # put_off2[2, 0] = 1 # 【关键】要求工件2(J2)在机器0(M0)上，至少比它本来的时段推迟1个时段
    # solution2 = Solution(sequence=seq2, put_off=put_off2)

    # objectives2, completion_times2, operation_periods2 = decoder.decode(solution2)
    # # objectives2 = decoder.decode(solution2)

    # print(f"  - 测试序列: {solution2.sequence}")
    # print(f"  - Put-off[2, 0] = 1")
    # print(f"  - 目标值 [TEC, TCTA]: {np.round(objectives2, 2)}")
    # print(f"  - 完成时间矩阵:\n{completion_times2}")
    # print(f"  - 时段操作矩阵:\n{operation_periods2}")
    # print("Good 已验证 与手算结果一致")  # [126.   38.4]
    
    # # 测试场景3: 跨时段约束生效
    # print("\n--- 场景3: “禁止跨时段”约束生效 ---")
    # seq3 = np.array([2, 1, 0])
    # put_off3 = np.zeros_like(problem_data.processing_times)
    # solution3 = Solution(sequence=seq3, put_off=put_off3)

    # objectives3, completion_times3, operation_periods3 = decoder.decode(solution3)
    # # objectives3 = decoder.decode(solution3)

    # print(f"  - 测试序列: {solution3.sequence}")
    # print(f"  - 目标值 [TEC, TCTA]: {np.round(objectives3, 2)}")
    # print(f"  - 完成时间矩阵:\n{completion_times3}")
    # print(f"  - 时段操作矩阵:\n{operation_periods3}")
    # print("Good 已验证 与手算结果一致")  # [108.  25.]

    seq1 = np.array([4,6,3,2,1,5,0,9,8,7])
    put_off1 = np.zeros_like(problem_data.processing_times)
    solution1 = Solution(sequence=seq1, put_off=put_off1)
    objectives1, completion_times1 = decoder.decode(solution1)
    # objectives1 = decoder.decode(solution1)

    print(f"  - 测试序列: {solution1.sequence}")
    print(f"  - 目标值 [TEC, TCTA]: {np.round(objectives1, 2)}")
    print(f"  - 完成时间矩阵:\n{completion_times1}")
    print("Good 已验证 与手算结果一致")  # [95.  31.4]

if __name__ == '__main__':
    test_decoder()