import numpy as np
import math

class BFO_for_benchmark:
    def __init__(self,
                 popsize: int,
                 N: int,            # 问题维度 (Dimension)
                 NC: int,
                 NR: int,
                 NM: int,
                 Mmax: int,
                 P1: float,
                 C: float,          # 趋向操作的步长 (Step size)
                 domain: tuple,     # 解的搜索范围, 例如 (-5.12, 5.12)
                 fitness_function):
        """
        初始化用于测试基准函数的细菌觅食算法

        参数:
        - popsize (int): 种群大小
        - N (int): 问题维度
        - NC (int): 趋向操作循环次数
        - NR (int): 复制操作循环次数
        - NM (int): 迁徙操作循环次数
        - Mmax (int): 最大趋向步数
        - P1 (float): 迁徙概率
        - C (float): 趋向步长
        - domain (tuple): 变量的取值范围 (min_val, max_val)
        - fitness_function (callable): 适应度函数
        """
        
        # BFO 算法参数
        self.popsize = popsize
        self.N = N
        self.NC = NC
        self.NR = NR
        self.NM = NM
        self.Mmax = Mmax
        self.P1 = P1
        self.C = C  # 新增：趋向步长
        self.domain = domain # 新增：搜索域
        
        # 适应度函数
        self.fitness_function = fitness_function

        # 初始化种群状态
        # MODIFIED: 位置直接是 N 维向量, 并在指定 domain 内初始化
        min_val, max_val = self.domain
        self.positions = np.random.uniform(min_val, max_val, size=(self.popsize, self.N))
        
        # 适应度和健康度
        self.fitness = np.full(self.popsize, np.inf)
        self.health = np.zeros(self.popsize)

        # 全局最优记录
        self.global_best_fitness = np.inf
        self.global_best_position = np.zeros(N) # MODIFIED: 记录最优位置而非序列
        
        # 初始化适应度
        self._evaluate_fitness()
        
    def _evaluate_fitness(self, indices_to_update=None):
        """
        MODIFIED: 直接根据位置评估适应度，去掉了序列解码
        """
        if indices_to_update is None:
            indices_to_update = range(self.popsize)
        
        for i in indices_to_update:
            # 直接将位置向量传入适应度函数
            self.fitness[i] = self.fitness_function(self.positions[i])

            # 更新全局最优
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_position = self.positions[i].copy()
                
    def _chemotaxis(self, current_step_size):
        """趋向操作"""
        for p in range(self.popsize):
            self.health[p] += self.fitness[p] # 健康度是适应度的累加
            
            last_fitness = self.fitness[p]

            for m in range(self.Mmax):
                position_old = self.positions[p].copy()
                fitness_old = self.fitness[p]

                # MODIFIED: 生成一个随机方向向量并单位化
                delta = np.random.uniform(-1, 1, self.N)
                delta_normalized = delta / np.linalg.norm(delta)
                
                # MODIFIED: 引入步长 C 进行移动
                self.positions[p] += current_step_size * delta_normalized
                
                # 边界处理: 如果超出定义域，则拉回到边界
                min_val, max_val = self.domain
                self.positions[p] = np.clip(self.positions[p], min_val, max_val)

                # 评估新位置
                self._evaluate_fitness(indices_to_update=[p])

                # 如果新位置更好，则继续朝这个方向“游泳”
                if self.fitness[p] < last_fitness:
                    last_fitness = self.fitness[p] # 更新“游泳”循环中的最优值
                else:
                    # 如果移动后没有变得更好，则结束本次“游泳”，下次将“翻滚”（产生新方向）
                    self.positions[p] = position_old # 还原到移动前的位置
                    self.fitness[p] = fitness_old
                    break # 退出 Mmax 循环，进行下一次翻滚

    def _reproduction(self):
        """复制操作"""
        sorted_indices = np.argsort(self.health)
        
        num_to_replace = self.popsize // 2
        best_indices = sorted_indices[:num_to_replace]
        worst_indices = sorted_indices[num_to_replace:]
        
        for i in range(num_to_replace):
            best_idx = best_indices[i]
            worst_idx = worst_indices[i]
            
            self.positions[worst_idx] = self.positions[best_idx].copy()
            self.fitness[worst_idx] = self.fitness[best_idx]
            
    def _migration(self):
        """
        MODIFIED: 迁徙操作改为随机重置
        """
        min_val, max_val = self.domain
        for p in range(self.popsize):
            if np.random.rand() < self.P1:
                self.positions[p] = np.random.uniform(min_val, max_val, size=self.N)
                self._evaluate_fitness(indices_to_update=[p])
            
    def run(self):
        """执行 BFO 算法"""
        print("BFO 算法开始运行...")
        c_initial = self.C  # 保存初始步长
        
        for l in range(self.NC):
            # 将健康度重置为0，用于在本次复制操作中重新计算
            current_c = c_initial * np.exp(-2.0 * (l / self.NC))
            self.health.fill(0) 
            
            for k in range(self.NR):
                for j in range(self.NM):
                    self._chemotaxis(current_step_size=current_c)
                self._reproduction()
            
            self._migration()
            
            print(f"第 {l+1}/{self.NC} 代完成, 当前全局最优适应度: {self.global_best_fitness:.6f}")
            
        print("\nBFO算法运行结束。")
        print(f"最终最优适应度: {self.global_best_fitness}")
        # print(f"最终最优解 (Position): {self.global_best_position}") # 如果需要可以取消注释
        return self.global_best_position, self.global_best_fitness
    
def sphere_function(x: np.ndarray) -> float:
    """Sphere 函数"""
    return np.sum(np.square(x))

def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin 函数"""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

BFO_PARAMS = {
    'popsize': 50,      # 种群大小
    'NC': 1000,           # 趋向次数
    'NR': 10,            # 复制次数
    'NM': 5,            # 迁徙次数
    'Mmax': 10,         # 趋向操作中的最大移动步数
    'P1': 0.25,         # 迁徙概率
    'C': 10,           # 游泳/翻滚的步长
}

# print("="*30)
# print("开始测试 Sphere 函数")
# print("="*30)
# sphere_bfo = BFO_for_benchmark(
#     **BFO_PARAMS,
#     N=10,                             # 测试10维问题
#     domain=(-100, 100),               # Sphere 函数的典型定义域
#     fitness_function=sphere_function
# )
# best_pos_sphere, best_fit_sphere = sphere_bfo.run()
# print(f"\nSphere 函数测试结果:")
# print(f"  - 最优适应度: {best_fit_sphere:.6e}") # 使用科学计数法显示，因为结果可能很小
# print(f"  - 已知全局最优: 0.0")


print("\n" + "="*30)
print("开始测试 Rastrigin 函数")
print("="*30)
rastrigin_bfo = BFO_for_benchmark(
    **BFO_PARAMS,
    N=10,                             # 测试10维问题
    domain=(-5.12, 5.12),              # Rastrigin 函数的典型定义域
    fitness_function=rastrigin_function
)
best_pos_rastrigin, best_fit_rastrigin = rastrigin_bfo.run()
print(f"\nRastrigin 函数测试结果:")
print(f"  - 最优适应度: {best_fit_rastrigin:.6f}")
print(f"  - 已知全局最优: 0.0")