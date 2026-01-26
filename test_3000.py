import time
import numpy as np
import sys
import os

# 确保能找到编译好的 kinsol_py 包
sys.path.append(os.path.dirname(__file__))
from pykinsol import kinsol

def main():
    # --- 1. 问题规模与参数设置 ---
    N = 3000
    epsilon = 0.1
    print(f"=== 测试场景: {N} 维 强耦合非线性方程组 ===")
    
    # --- 2. 构造目标真值 (Ground Truth) ---
    # 生成一个在 [0.5, 3.5] 之间的正弦波动真值
    indices = np.arange(N)
    x_true = 2.0 + 0.8 * np.sin(4 * np.pi * indices / N)
    
    # --- 3. 定义双边约束 ---
    lb = np.full(N, 0.5)
    ub = np.full(N, 3.5)

    # --- 4. 构造 RHS (使得 x_true 成为方程的精确解) ---
    # 方程形式: F_i = x_i^3 - epsilon * (x_{i-1} + x_{i+1}) - rhs_i = 0
    x_t_l = np.roll(x_true, 1); x_t_l[0] = 0.0
    x_t_r = np.roll(x_true, -1); x_t_r[-1] = 0.0
    rhs_vector = (x_true**3) - epsilon * (x_t_l + x_t_r)

    # --- 5. 定义残差函数 (物理空间的 N 维函数) ---
    def residual_func(x):
        # 注意: 这里的 x 是原始物理变量
        x_l = np.roll(x, 1); x_l[0] = 0.0
        x_r = np.roll(x, -1); x_r[-1] = 0.0
        return (x**3) - epsilon * (x_l + x_r) - rhs_vector

    # --- 6. 定义物理空间雅可比矩阵 ---
    def jacobian_func(x):
        # 构造物理方程的 N x N 雅可比矩阵
        # 内部 C++ 逻辑会自动构建如下结构的 2N 维矩阵: [JF, 0; I, I]
        J = np.zeros((N, N))
        diag_indices = np.arange(N)
        
        # 主对角线导数: dF_i / dx_i = 3 * x_i^2
        J[diag_indices, diag_indices] = 3.0 * (x**2)
        
        # 次对角线导数: dF_i / dx_{i-1} = dF_i / dx_{i+1} = -epsilon
        for i in range(N):
            if i > 0:   J[i, i-1] = -epsilon
            if i < N-1: J[i, i+1] = -epsilon
        return J

    # --- 7. 初始检查 ---
    # 初始猜想值在 [lb, ub] 内部
    x0 = np.full(N, 2.0) 
    print("-" * 30)
    print("=== 状态预检 ===")
    init_res = residual_func(x0)
    print(f"初始物理残差 Norm: {np.linalg.norm(init_res):.6e}")
    print(f"真值残差验证: {np.linalg.norm(residual_func(x_true)):.6e}")
    print("-" * 30)

    # --- 8. 调用求解器 ---
    print(f"=== 开始求解 (物理维度: {N}, 转换维度: {2*N}) ===")
    start_time = time.time()
    
    result = kinsol(
        func=residual_func,
        x0=x0,
        fprime=jacobian_func, 
        lb=lb,
        ub=ub
    )
    
    end_time = time.time()

    # --- 9. 结果展示与分析 ---
    if result["success"]:
        result_x = result["x"]
        duration_ms = (end_time - start_time) * 1000
        
        print(f"求解成功!")
        print(f"总耗时: {duration_ms:.3f} ms")
        print(f"最终解 x: {result_x}")
        print(f"最终残差 Norm: {result['fun']:.3e}")

        manual_res_vector = residual_func(result_x)
        manual_norm = np.linalg.norm(manual_res_vector)
        print(f"手动计算验证: {manual_norm:.3e}")

        
        # 边界约束验证
        within_bounds = np.all((result_x >= lb - 1e-9) & (result_x <= ub + 1e-9))
        print(f"约束满足检查: {within_bounds}")
    else:
        print(f"求解失败。状态码: {result.get('status')}, 最终残差: {result['fun']:.3e}")

if __name__ == "__main__":
    main()