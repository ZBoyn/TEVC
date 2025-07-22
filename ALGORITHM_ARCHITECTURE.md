# 算法架构与实现文档

本文档详细阐述了当前多目标优化算法的整体架构、核心组件和执行流程。

## 1. 整体架构

算法采用了一种**两阶段混合进化策略**，以经典的 **NSGA-II** 算法为骨架，并集成了针对本问题的特定启发式算子。其核心思想是将广泛的全局探索和深度的局部优化在不同阶段进行，以平衡求解效率和解的质量。

### 1.1. 核心模块

项目被高度模块化，主要包括以下几个部分：

-   `config.py`: 定义了问题的核心数据结构，特别是 `ProblemDefinition` 和 `Solution` 类。这是整个算法的数据基础。
-   `data_loader.py`: 负责从文本文件加载问题实例，并填充 `ProblemDefinition` 对象。
-   `main_algorithm.py`: 包含了算法的主控制流程 `EvolutionaryAlgorithm`，负责种群初始化、进化循环、阶段判断和最终选择。
-   `operators.py`: 封装了所有的进化算子，分为 `BFO_Operators` 和 `LocalSearch_Operators` 两大类，是产生新解的核心。
-   `decode.py`: 实现了`Decoder`类，负责将一个 `Solution` 对象（即一个调度方案）翻译成两个目标函数值（TEC 和 TCTA），是评估解优劣的唯一标准。
-   `results_handler.py`: 负责将算法最终找到的帕累托最优解集保存为 CSV 文件和PNG图像。

### 1.2. 解的表示 (`Solution` Class)

一个解由两部分核心数据定义：
1.  `sequence`: 一个一维Numpy数组，表示工件的加工顺序。
2.  `final_schedule`: 一个二维Numpy矩阵，存储每个工序**最终的绝对完成时间**。该矩阵由 `right_shift` 算子生成。如果一个解未经 `right_shift` 优化，则此值为 `None`。

我们已经**废除了**早期版本中不稳定的 `put_off` 相对延迟机制。

## 2. 算法流程伪代码

算法的执行流程可以被清晰地划分为两个阶段。

### 2.1. 主流程 (`EvolutionaryAlgorithm.run`)

```
Function Run_Evolutionary_Algorithm:
  Input: 问题定义, 种群大小, 总代数, 精修阶段起始代数
  Output: 帕累托最优解集 (Archive)

  1.  // 初始化
  2.  Population = Initialize_Population() // 生成初始种群 (均为紧凑调度)
  3.  Evaluate(Population) // 使用解码器评估所有初始解
  4.  Archive = Update_Archive(Population) // 将初始非支配解存入存档

  5.  // 主进化循环
  6.  For gen from 1 to Max_Generations:
  7.      // 阶段判断
  8.      If gen >= Polishing_Phase_Start_Gen:
  9.          // B. 精修阶段
  10.         Offspring = Polish_Population(Population)
  11.     Else:
  12.         // A. 常规进化阶段
  13.         Offspring = Generate_Offspring(Population)
  14.
  15.     // 评估与选择 (NSGA-II核心)
  16.     Evaluate(Offspring)
  17.     Archive = Update_Archive(Offspring)
  18.     Combined_Population = Population + Offspring
  19.     Fronts = Non_Dominated_Sort(Combined_Population)
  20.     Population = Select_Next_Generation(Fronts, Population_Size)
  21.
  22. // 循环结束
  23. Return Archive
```

### 2.2. 常规进化阶段 (`Generate_Offspring`)

此阶段的目标是使用速度较快的算子广泛探索解空间。

```
Function Generate_Offspring(Population):
  1.  New_Offspring = []
  2.  While size of New_Offspring < Population_Size:
  3.      Select Parent(s) from Population using Tournament_Selection
  4.
  5.      // 概率性地选择一个算子
  6.      If random() < P_crossover:
  7.          Child1, Child2 = Crossover(Parent1, Parent2) // 仅交叉sequence
  8.          // Child1, Child2的final_schedule均为None
  9.          Add Child1, Child2 to New_Offspring
  10.     Else if random() < P_chemotaxis:
  11.         Child = Chemotaxis(Parent) // 序列变异
  12.         Add Child to New_Offspring
  13.     Else if random() < P_prefer_agent:
  14.         Child = Prefer_Agent_Operator(Parent) // 针对TCTA的启发式序列优化
  15.         Add Child to New_Offspring
  16.     Else if random() < P_right_shift:
  17.         // 注意: 此阶段的right_shift用于探索, 它会将一个紧凑调度变为带延迟的优化调度
  18.         Child = Right_Shift_Operator(Parent)
  19.         Add Child to New_Offspring
  20.
  21. Return New_Offspring
```

### 2.3. 精修阶段 (`Polish_Population`)

此阶段的目标是使用强大的、高计算成本的组合算子对当前最优的种群进行深度优化。

```
Function Polish_Population(Population):
  1.  Polished_Offspring = []
  2.  For each Parent_Solution in Population:
  3.      // 第一步: NEH专家进行序列重构, 追求极致TCTA
  4.      // 该操作会返回一个final_schedule=None的紧凑调度解
  5.      Compact_Solution = Destroy_Rebuild_Operator(Parent_Solution)
  6.
  7.      // 第二步: Right_Shift专家在新的优秀序列上, 追求极致TEC
  8.      // 该操作会为解赋予一个精确的final_schedule
  9.      Final_Polished_Solution = Right_Shift_Operator(Compact_Solution)
  10.
  11.     Add Final_Polished_Solution to Polished_Offspring
  12.
  13. Return Polished_Offspring
``` 