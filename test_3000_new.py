import time
import numpy as np
import sys
import os

# ==========================================
# 【关键修改 1】 导入方式适配
# 您的包现在是一个顶层 pyd，叫 pykinsol
# ==========================================
try:
    import pykinsol
    print(f"✅ 成功导入 pykinsol 模块")
    print(f"📂 模块路径: {pykinsol.__file__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请检查是否已执行 pip install . 并且不在源码目录下运行此脚本")
    sys.exit(1)

def main():
    # --- 1. 问题规模与参数设置 ---
    # 3000 维对于 Dense 矩阵来说有点大 (9百万个元素)，GMRES 会快很多
    N = 3000
    epsilon = 0.1
    print(f"\n=== 测试场景: {N} 维 强耦合非线性方程组 ===")

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
    print(f"初始物理残差 Norm: {np.linalg.norm(residual_func(x0)):.6e}")
    print("-" * 30)

    # =================================================================
    # 测试 1: GMRES 求解器 (推荐用于大系统)
    # =================================================================
    print(f"\n>>> [测试 1] 正在使用 [GMRES + LineSearch] 策略求解...")
    start_time = time.time()
    
    # 【关键修改 2】 调用方式适配: pykinsol.pykinsol(...)
    result_gmres = pykinsol.pykinsol(
        func=residual_func,
        x0=x0,
        fprime=None,       # GMRES 模式不需要 Jacobian，传 None
        lb=lb, 
        ub=ub,
        method='linesearch',
        linear_solver='gmres'  # 使用您新增的 GMRES 功能
    )
    
    duration = (time.time() - start_time) * 1000
    print(f"GMRES 耗时: {duration:.3f} ms")
    print(f"GMRES 结果状态: {'成功' if result_gmres['success'] else '失败'}")
    print(f"GMRES 最终残差: {result_gmres['fun']:.3e}")

    # =================================================================
    # 测试 2: Dense 求解器 (旧模式，用于对比)
    # =================================================================
    print(f"\n>>> [测试 2] 正在使用 [Dense + LineSearch] 策略求解...")
    start_time = time.time()
    
    # 注意: 3000维 Dense 矩阵约 72MB，计算稍慢是正常的
    result_dense = pykinsol.pykinsol(
        func=residual_func,
        x0=x0,
        fprime=jacobian_func,    # Dense 模式必须提供 Jacobian
        lb=lb, 
        ub=ub,
        method='linesearch',
        linear_solver='dense'
    )
    
    duration = (time.time() - start_time) * 1000
    print(f"Dense 耗时: {duration:.3f} ms")
    print(f"Dense 结果状态: {'成功' if result_dense['success'] else '失败'}")
    print(f"Dense 最终残差: {result_dense['fun']:.3e}")

    # =================================================================
    # 结果验证 (以 GMRES 结果为例)
    # =================================================================
    if result_gmres["success"]:
        final_x = result_gmres["x"]
        
        # 边界约束验证
        within_bounds = np.all((final_x >= lb - 1e-9) & (final_x <= ub + 1e-9))
        print(f"\n>>> 约束满足检查: {within_bounds}")
        
        # 精度检查
        err = np.linalg.norm(final_x - x_true)
        print(f">>> 与真值误差 Norm: {err:.3e}")
        
        if err < 1e-4:
            print("🎉 测试通过！求解结果非常精确。")
        else:
            print("⚠️ 警告：虽然收敛但精度似乎一般，请检查物理模型。")

if __name__ == "__main__":
    main()