import time
import numpy as np
import sys
import os
# 确保能找到编译好的 kinsol_py 包
# sys.path.append(os.path.dirname(__file__)) # 如果已经pip install了就不需要这行
from pykinsol import pykinsol

def main():
    # --- 1. 问题规模与参数设置 ---
    N = 3000
    epsilon = 0.1
    print(f"=== 测试场景: {N} 维 强耦合非线性方程组 ===")

    # --- 2. 构造目标真值 (Ground Truth) ---
    indices = np.arange(N)
    x_true = 2.0 + 0.8 * np.sin(4 * np.pi * indices / N)

    # --- 3. 定义双边约束 ---
    lb = np.full(N, 0.5)
    ub = np.full(N, 3.5)

    # --- 4. 构造 RHS ---
    x_t_l = np.roll(x_true, 1); x_t_l[0] = 0.0
    x_t_r = np.roll(x_true, -1); x_t_r[-1] = 0.0
    rhs_vector = (x_true**3) - epsilon * (x_t_l + x_t_r)

    # --- 5. 定义残差函数 ---
    def residual_func(x):
        x_l = np.roll(x, 1); x_l[0] = 0.0
        x_r = np.roll(x, -1); x_r[-1] = 0.0
        return (x**3) - epsilon * (x_l + x_r) - rhs_vector

    # --- 6. 定义雅可比矩阵 (仅供 Dense 模式使用) ---
    def jacobian_func(x):
        # 注意：对于 3000x3000 矩阵，Dense 模式其实很慢且耗内存
        J = np.zeros((N, N))
        diag_indices = np.arange(N)
        J[diag_indices, diag_indices] = 3.0 * (x**2)
        for i in range(N):
            if i > 0:   J[i, i-1] = -epsilon
            if i < N-1: J[i, i+1] = -epsilon
        return J

    # --- 7. 初始检查 ---
    x0 = np.full(N, 2.0) 
    print("-" * 30)
    print("=== 状态预检 ===")
    init_res = residual_func(x0)
    print(f"初始物理残差 Norm: {np.linalg.norm(init_res):.6e}")
    print("-" * 30)

    # =================================================================
    # 【核心修改区域】 适配新接口调用
    # =================================================================
    
    # --- 方案 1: 使用 GMRES (强烈推荐用于 3000 维问题) ---
    # 优点: 极快，不需要构建巨大的 Jacobian 矩阵
    # 缺点: 雅可比信息通过差分获得
    # print(f"\n>>> 正在使用 [GMRES + LineSearch] 策略求解...")
    # start_time = time.time()
    
    # result = pykinsol(
    #     func=residual_func,
    #     x0=x0,
    #     fprime=None,         # GMRES 模式通常不需要显式 Jacobian
    #     lb=lb, 
    #     ub=ub,
    #     method='linesearch',     # 策略: 'linesearch' 或 'newton'
    #     linear_solver='gmres'    # 求解器: 'gmres' (稀疏/大系统) 或 'dense'
    # )

    
    # --- 方案 2: 使用 Dense (你之前的逻辑) ---
    print(f"\n>>> 正在使用 [Dense + LineSearch] 策略求解...")
    start_time = time.time()
    result = pykinsol(
        func=residual_func,
        x0=x0,
        fprime=jacobian_func,    # Dense 模式必须提供 Jacobian 以加速
        lb=lb, 
        ub=ub,
        method='linesearch',
        linear_solver='dense'    # 使用稠密矩阵求解
    )
    
    end_time = time.time()
    # =================================================================

    # --- 9. 结果展示与分析 ---
    duration_ms = (end_time - start_time) * 1000
    
    if result["success"]:
        result_x = result["x"]
        
        print(f"求解成功!")
        print(f"总耗时: {duration_ms:.3f} ms")
        # print(f"最终解 x: {result_x}") # 3000维太长，不打印了
        
        # 此时 result['fun'] 已经是你修改 C++ 后返回的纯物理残差了
        print(f"最终残差 Norm (from Solver): {result['fun']:.3e}")

        # 手动计算验证
        manual_res_vector = residual_func(result_x)
        manual_norm = np.linalg.norm(manual_res_vector)
        print(f"手动计算验证残差值: {manual_norm:.3e}")

        # 边界约束验证
        within_bounds = np.all((result_x >= lb - 1e-9) & (result_x <= ub + 1e-9))
        print(f"约束满足检查: {within_bounds}")

    else:
        print(f"求解失败。状态码: {result.get('status')}, 最终残差: {result['fun']:.3e}")

if __name__ == "__main__":
    main()