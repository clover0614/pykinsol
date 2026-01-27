import numpy as np
from . import kinsol_cpp

# 暴露 C++ 定义的常量
KIN_NONE = kinsol_cpp.KIN_NONE
KIN_LINESEARCH = kinsol_cpp.KIN_LINESEARCH

def kinsol(func, x0, fprime=None, lb=None, ub=None, 
           method='linesearch', linear_solver='dense'):
    """
    Python 接口封装
    
    Parameters
    ----------
    method : str, optional
        'linesearch' (默认) - 使用牛顿法 + 线搜索 (更稳健)
        'newton' - 纯牛顿法 (收敛快，但要求初值好)
        
    linear_solver : str, optional
        'dense' (默认) - 稠密直接求解 (SUNDenseMatrix). 适合 N < 5000.
        'gmres' - 迭代求解 (SPGMR). 适合大规模稀疏系统 N > 5000.
                  在此模式下，fprime (Jacobian) 可能会被忽略，使用内部差分近似 J*v.
    """
    
    # 1. 参数预处理
    x_input = np.ascontiguousarray(x0, dtype=np.float64)
    N = x_input.size
    
    if lb is None: lb = np.full(N, -np.inf)
    else: lb = np.ascontiguousarray(lb, dtype=np.float64)
        
    if ub is None: ub = np.full(N, np.inf)
    else: ub = np.ascontiguousarray(ub, dtype=np.float64)
        
    x_input = np.clip(x_input, lb + 1e-8, ub - 1e-8)
    
    # 2. 策略映射
    strategy_map = {
        'linesearch': kinsol_cpp.KIN_LINESEARCH,
        'newton': kinsol_cpp.KIN_NONE
    }
    strategy_int = strategy_map.get(method.lower(), kinsol_cpp.KIN_LINESEARCH)

    # 3. 线性求解器映射
    linsol_map = {
        'dense': kinsol_cpp.LINSOL_DENSE,
        'gmres': kinsol_cpp.LINSOL_GMRES
    }
    linsol_int = linsol_map.get(linear_solver.lower(), kinsol_cpp.LINSOL_DENSE)

    # 4. 调用底层 C++ 扩展
    result = kinsol_cpp.solve(func, x_input, fprime, lb, ub, strategy_int, linsol_int)
    
    return result

__all__ = ["kinsol", "KIN_NONE", "KIN_LINESEARCH"]