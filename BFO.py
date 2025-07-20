import numpy as np
import math

class BFO:
    def __init__(self,
                 C_initial: float, 
                 C_final: float,
                 popsize: int,
                 N: int,
                 NC: int,
                 NR: int,
                 NM: int,
                 Mmax: int,
                 P1: float,
                 fitness_function,
                 **problem_params):
        """
        初始化细菌觅食算法

        参数:
        - popsize (int): 种群大小 (细菌数量)
        - N (int): 问题维度 (工件数量)
        - NC (int): 趋向操作循环次数
        - NR (int): 复制操作循环次数
        - NM (int): 迁徙操作循环次数
        - Mmax (int): 最大趋向步数 (翻滚/游泳最大次数)
        - P1 (float): 迁徙概率
        - fitness_function (callable): 适应度函数
        - problem_params: 适应度函数需要的额外参数 (例如: 加工时间、电价等)
        """
        
        # BFO 算法参数
        self.popsize = popsize
        self.N = N
        self.NC = NC
        self.NR = NR
        self.NM = NM
        self.Mmax = Mmax
        self.P1 = P1
        self.C_initial = C_initial
        self.C_final = C_final
        self.total_iterations = NC * NR * NM
        
        # 适应度函数及其参数
        self.fitness_function = fitness_function
        self.problem_params = problem_params

        # 初始化种群状态
        # 位置: 随机初始化在 [-1, 1] x [-1, 1] 的空间内
        self.positions = np.random.uniform(-1, 1, size=(self.popsize, self.N, 2))
        
        # 序列、适应度和健康度
        self.sequences = np.zeros((self.popsize, self.N), dtype=int)
        self.fitness = np.full(self.popsize, np.inf)  # 初始化为无穷大, 表示未评估
        self.tec_values = np.full(self.popsize, np.inf)
        self.tcta_values = np.full(self.popsize, np.inf)
        self.health = np.zeros(self.popsize)          # 初始化为0, 表示未健康

        # 全局最优记录
        self.global_best_fitness = np.inf
        self.global_best_sequence = np.zeros(self.N, dtype=int)
        
        # 初始化序列和适应度
        self._update_sequences_and_fitness()
        
    def _update_sequences_and_fitness(self, indices_to_update=None):
        """根据位置更新序列和适应度"""
        if indices_to_update is None:
            indices_to_update = range(self.popsize)
        
        for i in indices_to_update:
            # 解码：从坐标 (positions) 生成序列 (sequences)
            # 计算每个工件坐标到原点的距离
            distances = np.linalg.norm(self.positions[i], axis=1)
            # 根据距离排序, 得到工件序列
            self.sequences[i] = np.argsort(distances)
            
            # 评估
            self.fitness[i] = self.fitness_function(self.sequences[i], **self.problem_params)

            # 更新全局最优
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_sequence = self.sequences[i].copy()

    def _chemotaxis(self, current_step_size):
        """趋向操作"""
        for p in range(self.popsize):
            # 保存个体 p 在趋向开始前的适应度, 用于累计健康度
            self.health[p] = self.fitness[p]
            
            # 初始化随机方向
            # flag=1 代表上一步是成功的，可以继续朝原方向“游泳”
            # flag=0 代表需要重新选择方向“翻滚”
            flag = 0
            deta = np.zeros((self.N, 2))
            
            for m in range(self.Mmax):
                fitness_old = self.fitness[p]  # 保存当前适应度
                sequence_old = self.sequences[p].copy()  # 保存当前序列
                position_old = self.positions[p].copy()  # 保存当前位置信息

                # 生成随机方向
                if flag == 0:
                    theta = np.random.uniform(0, 2 * math.pi, self.N)
                    deta[:, 0] = np.cos(theta)
                    deta[:, 1] = np.sin(theta)

                # 移动到新的位置
                self.positions[p] += current_step_size * deta
                
                # 从新位置解码出新的序列 并计算适应度
                self._update_sequences_and_fitness(indices_to_update=[p])

                # 如果新适应度更好, 则保留新位置
                if self.fitness[p] < fitness_old:
                    # 移动成功 保持原方向
                    flag = 1
                else:
                    # 移动失败 还原位置和序列 准备下次翻滚
                    flag = 0
                    self.positions[p] = position_old
                    self.sequences[p] = sequence_old
                    self.fitness[p] = fitness_old
                    
    def _reproduction(self):
        """复制操作"""
        # 根据健康度对个体进行排序
        # 在我们的最小化问题中, Health值越小, 代表个体越“健康”（历史表现越好）
        sorted_indices = np.argsort(self.health) # [最优个体的索引, 次优个体索引, ..., 最差个体索引]
        
        # 取一半个题进行淘汰和复制
        num_to_replace = self.popsize // 2
        best_indices = sorted_indices[:num_to_replace]  # 取健康度最小, 即最优的一半: 前一半
        worst_indices = sorted_indices[num_to_replace:]
        
        # 用最优的一半个题替换最差的一半
        for i in range(num_to_replace):
            best_idx = best_indices[i]
            worst_idx = worst_indices[i]
            
            self.positions[worst_idx] = self.positions[best_idx].copy()
            self.sequences[worst_idx] = self.sequences[best_idx].copy()
            self.fitness[worst_idx] = self.fitness[best_idx]
            
    def _migration(self, strategy='generate_new', w_tcta=0.5, w_tec=0.5):
        """自适应迁徙操作

        Args:
            strategy (str): 迁徙时的替换策略。
                        'generate_new' (默认): 生成全新随机坐标，探索能力强 (推荐)。
                        'shuffle': 打乱个体当前坐标，进行局部变异。
        w_tcta (float): TCTA(总工时)在计算迁徙概率时的权重。
        w_tec (float): TEC(总电费)在计算迁徙概率时的权重。
        """
        
        # 计算出整个种群的目标值范围
        epsilon = 1e-9
        tcta_max = self.tcta_values.max()
        tcta_min = self.tcta_values.min()
        tcta_range = tcta_max - tcta_min + epsilon
        
        tec_max = self.tec_values.max()
        tec_min = self.tec_values.min()
        tec_range = tec_max - tec_min + epsilon
        
        # 计算每个个体归一化的性能 越高表示越差
        norm_tcta = (self.tcta_values - tcta_min) / tcta_range
        norm_tec = (self.tec_values - tec_min) / tec_range
        
        # 计算每个个体的"差度" 作为迁徙概率
        mc = w_tcta * norm_tcta + w_tec * norm_tec
        
        # 根据概率决定哪些个体需要迁徙
        random_trigger = np.random.rand(self.popsize)
        migration_indices = np.where(random_trigger < mc)[0]
        
        # 如果有个体需要迁徙
        if len(migration_indices) > 0:
            # print(f"迁徙个体数量: {len(migration_indices)}")
            for i in migration_indices:
                if strategy == 'generate_new':
                    # 生成全新随机坐标
                    self.positions[i] = np.random.uniform(-1, 1, size=(self.N, 2))
                elif strategy == 'shuffle':
                    # 打乱当前坐标
                    indices = np.arange(self.N)
                    np.random.shuffle(indices)
                    self.positions[i] = self.positions[i][indices]
                else:
                    raise ValueError("Unsupported migration strategy. Use 'generate_new' or 'shuffle'.")

            self._update_sequences_and_fitness(indices_to_update=migration_indices)
            
    def run(self):
        """执行 BFO 算法"""
        current_iter = 0
        for l in range(self.NC): # 趋向操作循环
            for j in range(self.NR):
                self.health.fill(0)
                
                for k in range(self.NM):
                    progress = current_iter / self.total_iterations
                    current_C = self.C_initial - (self.C_initial - self.C_final) * progress
                    
                    self._chemotaxis(current_step_size=current_C)
                    current_iter += 1
                    # print(f"  (NC={l+1}, NR={j+1}, NM={k+1}) -> Current Best Fitness: {self.global_best_fitness}")
            
                self._reproduction()  # 复制操作
            self._migration(strategy='generate_new')  # 迁徙操作

            print(f"第 {l+1} 代完成, 当前全局最优适应度: {self.global_best_fitness}, 最优序列: {self.global_best_sequence}")
            
        print("\nBFO算法运行结束。")
        print(f"最终最优适应度: {self.global_best_fitness}")
        print(f"最终最优工件序列: {self.global_best_sequence}")
        return self.global_best_sequence, self.global_best_fitness