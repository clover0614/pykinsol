# 上层封装，定义用户调用的接口设计
import numpy as np
from . import kinsol_cpp

def kinsol(func, x0, fprime=None, lb=None, ub=None):
    """
    Python 接口封装：处理参数预处理与边界初始化 
    """
    # 1. 确保输入为双精度浮点数且内存连续
    x_input = np.ascontiguousarray(x0, dtype=np.float64)
    N = x_input.size
    
    # 2. 边界条件初始化
    if lb is None:
        lb = np.full(N, -np.inf) # 使用 inf 更符合数学习惯
    else:
        lb = np.ascontiguousarray(lb, dtype=np.float64)
        
    if ub is None:
        ub = np.full(N, np.inf)
    else:
        ub = np.ascontiguousarray(ub, dtype=np.float64)
        
    # 3. 边界安全性检查与修正 
    # 确保初始猜想值严格在内部区域，不在边界上
    x_input = np.clip(x_input, lb + 1e-8, ub - 1e-8)
    
    # 4. 调用底层 C++ 扩展 [cite: 12, 64]
    # fprime 如果为 None，C++ 内部应处理为使用差分法
    result = kinsol_cpp.solve(func, x_input, fprime, lb, ub)
    
    return result

# 定义包级别公开的接口
__all__ = ["kinsol"]