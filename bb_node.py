import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import ast
import os

@dataclass(order=False)
class SearchNode:
    """定义搜索树中的一个节点，代表一个部分调度方案"""
    sequence: Tuple[int, ...] = field(default_factory=tuple)
    put_off_decisions: Tuple[int, ...] = field(default_factory=tuple)
    unscheduled_jobs: Set[int] = field(default_factory=set)
    completion_times_m1: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_bound_tec: float = 0.0
    lower_bound_tcta: float = 0.0
    priority: float = 0.0
    
    def __lt__(self, other: 'SearchNode') -> bool:
        return self.priority < other.priority