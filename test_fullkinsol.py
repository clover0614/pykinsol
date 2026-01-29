import time
import numpy as np
import sys
import os

# ==========================================
# 1. 导入检查
# ==========================================
try:
    # 尝试导入我们刚编译好的模块
    import pykinsol
    print(f"✅ 成功导入 pykinsol 模块")
    print(f"📂 模块路径: {pykinsol.__file__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已编译 C++ 扩展 (python setup.py build_ext --inplace)")
    sys.exit(1)

# 从模块中导入主函数
from pykinsol import pykinsol

def main():
    # --- 1. 问题设置 (3000维非线性方程组) ---
    N = 3000
    epsilon = 0.1 # 耦合强度
    print(f"\n=== 测试场景: {N} 维 强耦合非线性方程组 ===")

    # --- 2. 构造真值 (Ground Truth) ---
    indices = np.arange(N)
    x_true = 2.0 + 0.8 * np.sin(4 * np.pi * indices / N)

    # --- 3. 定义边界 (模拟双边约束) ---
    lb = np.full(N, 0.5)
    ub = np.full(N, 3.5)

    # --- 4. 构造 RHS (使得 x_true 是精确解) ---
    x_t_l = np.roll(x_true, 1); x_t_l[0] = 0.0
    x_t_r = np.roll(x_true, -1); x_t_r[-1] = 0.0
    rhs_vector = (x_true**3) - epsilon * (x_t_l + x_t_r)

    # --- 5. 定义残差函数 (包含 Clipping 逻辑) ---
    def residual_func(x):
        # 【关键演示】: 
        # 在 N 维无松弛方案中，KINSOL 可能会尝试超出 lb/ub 的 x。
        # 我们在这里做“软截断”或“投影”，保证物理计算不崩。
        # 对于 x^3 这种数学函数其实不需要，但对于 Log/Exp 物理模型必须有。
        x_safe = np.clip(x, lb, ub) 
        
        # 计算残差
        x_l = np.roll(x_safe, 1); x_l[0] = 0.0
        x_r = np.roll(x_safe, -1); x_r[-1] = 0.0
        
        res = (x_safe**3) - epsilon * (x_l + x_r) - rhs_vector
        return res

    # --- 6. 定义解析雅可比 (仅 Dense 模式需要) ---
    def jacobian_func(x):
        # 注意：这里返回的是 N x N 矩阵，不再是 2N x 2N
        J = np.zeros((N, N))
        diag_indices = np.arange(N)
        
        # 简单的带状矩阵
        x_safe = np.clip(x, lb, ub)
        J[diag_indices, diag_indices] = 3.0 * (x_safe**2)
        
        for i in range(N):
            if i > 0:   J[i, i-1] = -epsilon
            if i < N-1: J[i, i+1] = -epsilon
        return J

    # --- 7. 初始猜测 ---
    x0 = np.full(N, 1.5) # 离真值稍微远一点
    
    print("-" * 50)
    print(f"初始残差 Norm: {np.linalg.norm(residual_func(x0)):.6e}")
    print("-" * 50)

    # =================================================================
    # 测试 A: GMRES 求解器 (新接口重点测试)
    # =================================================================
    print(f"\n>>> [测试 A] GMRES + 差分雅可比 + 日志开启")
    print("    预期：速度快，内存占用小，能看到 KINSOL 内部迭代日志")
    
    start_time = time.time()
    
    result_gmres = pykinsol(
        func=residual_func,
        x0=x0,
        fprime=None,         # 【关键】GMRES 不需要 Python 雅可比，传 None 触发内部差分
        lb=lb,               # 传入 lb/ub 仅供参考，实际约束在 func 内 clip
        ub=ub,
        method='linesearch',
        linear_solver='gmres', # 【关键】指定 GMRES
        verbose=1            # 【关键】开启日志：观察 pnorm 和 fnorm
    )
    
    duration = (time.time() - start_time) * 1000
    print(f"GMRES 耗时: {duration:.3f} ms")
    print(f"GMRES 状态: {result_gmres['status']} ({'成功' if result_gmres['success'] else '失败'})")
    print(f"GMRES 残差: {result_gmres['fun']:.3e}")

    # =================================================================
    # 测试 B: Dense 求解器 (回归测试)
    # =================================================================
    print(f"\n>>> [测试 B] Dense + 解析雅可比 + 静默模式")
    print("    预期：速度较慢(N=3000)，但只要 Jacobian 写对了一定能收敛")
    
    start_time = time.time()
    
    result_dense = pykinsol(
        func=residual_func,
        x0=x0,
        fprime=jacobian_func, # 【关键】Dense 模式必须传 Jacobian
        lb=lb, 
        ub=ub,
        method='linesearch',
        linear_solver='dense',
        verbose=0             # 关闭日志
    )
    
    duration = (time.time() - start_time) * 1000
    print(f"Dense 耗时: {duration:.3f} ms")
    print(f"Dense 状态: {result_dense['status']} ({'成功' if result_dense['success'] else '失败'})")
    print(f"Dense 残差: {result_dense['fun']:.3e}")

    # =================================================================
    # 结果验证
    # =================================================================
    if result_gmres["success"]:
        final_x = result_gmres["x"]
        err = np.linalg.norm(final_x - x_true)
        print(f"\n>>> 结果验证 (与真值对比):")
        print(f"    L2 误差: {err:.3e}")
        if err < 1e-4:
            print("🎉 测试通过！结果精确。")
        else:
            print("⚠️ 精度不足。")
    else:
        print("\n❌ GMRES 求解失败，请检查日志。")

if __name__ == "__main__":
    main()