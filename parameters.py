CONFIG = {
    'DATA_FILE_PATH': "data2/3M10N-5.txt",  # 数据文件路径
    # 'DATA_FILE_PATH': "dataset/data_A3_J400_M4_1.txt",
    'PLOT_PARAMS': {
        'plot_frequency': 10, # 绘图频率
    },
    'POP_SIZE': 50, # 种群大小
    'MAX_GENERATIONS': 500, # 最大代数
    'INIT_PARAMS': {
        'h1_count': 1, # 启发式1的解数量
        'h2_count': 1, # 启发式2的解数量
        'mutation_swaps': 30, # 变异交换次数
        'agent_tca_estimation_samples': 10, # 代理TCA估计样本数
        'random_init_ratio': 1/3, # 随机初始化比例
    },

    'BFO_PARAMS': {
        'Mmax': 10,  # 最大步数
        'C_initial': 0.1, # 初始步长
        'C_final': 0.01, # 最终步长
        'put_off_mutation_prob': 0.5, # 偏移量变异概率
        'put_off_mutation_strength': 5, # 偏移量变异强度(有多少元素会发生改变)
        'put_off_regression_prob': 0.7, # 偏移量回归概率(有多少元素会回归0)
        'migration_tec_weight': 0.5, # 迁移TEC权重
        'migration_tcta_weight': 0.5, # 迁移TCTA权重
    },
    
    'PROB_PARAMS': {
        'prob_crossover': 0.9, # 交叉概率
        'prob_chemotaxis': 0.3, # 趋化概率
        'prob_prefer_agent': 0.2, # 偏好代理概率
        'prob_right_shift': 0.2, # 右移概率
        'polishing_phase_gens': 30, # 精修阶段代数
        'destroy_rebuild_alpha': 0.5, # 破坏重建比例
        'prob_polish': 0.4, # 在精修阶段应用强力局部搜索的概率
        'prob_mutation': 0.2, # 常规进化中应用变异算子的概率
        'migration_freq': 50, # 迁徙频率
    }
}