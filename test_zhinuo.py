# 初值设置的不好，会导致牛顿方向与实际解相反，最终收敛到边界上而停止迭代
import time
import numpy as np
import sys
import os

# 确保能找到编译好的 kinsol_py 包
sys.path.append(os.path.dirname(__file__))
from pykinsol import kinsol

def main():
    # --- 1. 问题规模与参数设置 ---
    N = 100 
    print(f"=== 测试场景: {N} 维 局部极值导致的“撞墙”死锁 ===")
    print("原理: 真解在 2.0，但初值在局部山坡上，导数强制将点推向左侧界外")
    
    # --- 2. 定义约束范围 ---
    lb = np.full(N, 0.5)
    ub = np.full(N, 3.5)

    # --- 3. 定义残差函数 F(x) ---
    # 我们构造 F(x) = (x - 2.0) * (x - 0.2) * (x + 1.0)
    # 这是一个三次函数，在可行域 [0.5, 3.5] 内唯一的根是 2.0
    # 但在 0.5 到 1.0 之间，函数斜率会引导点向左走（去寻找 0.2 或 -1.0）
    def residual_func(x):
        # F(x) = (x - 2.0)(x - 0.2)(x + 1.0)
        # 展开或直接相乘
        return (x - 2.0) * (x - 0.2) * (x + 1.0)

    # --- 4. 定义雅可比矩阵 ---
    def jacobian_func(x):
        # F'(x) = 3x^2 - 2.4x - 1.4 (根据导数公式计算)
        J = np.zeros((N, N))
        diag_val = 3.0 * (x**2) - 2.4 * x - 1.4
        np.fill_diagonal(J, diag_val)
        return J

    # --- 5. 初始检查 ---
    # 关键：初值选在 0.8。
    # 在 0.8 处，F(0.8) = (0.8-2)*(0.8-0.2)*(0.8+1) = -1.2 * 0.6 * 1.8 = -1.296
    # 导数 J(0.8) = 3(0.64) - 2.4(0.8) - 1.4 = 1.92 - 1.92 - 1.4 = -1.4
    # 牛顿步 = -F/J = -(-1.296)/(-1.4) = -0.925
    # 下一步预判点 = 0.8 - 0.925 = -0.125 (这就在界外了！)
    x0 = np.full(N, 0.8) 
    
    print("-" * 30)
    print("=== 状态预检 ===")
    print(f"下界约束 lb: {lb[0]}")
    print(f"理想目标解: 2.0 (在界内)")
    print(f"初始点 x0: {x0[0]}")
    
    # 计算初始步的方向
    f0 = residual_func(x0)[0]
    j0 = jacobian_func(x0)[0, 0]
    step = -f0 / j0
    print(f"初始牛顿建议步长: {step:.4f} -> 目标指向: {x0[0] + step:.4f} (界外)")
    print("-" * 30)

    # --- 6. 调用求解器 ---
    print(f"=== 开始求解 (牛顿法将尝试翻墙失败) ===")
    start_time = time.time()
    
    result = kinsol(
        func=residual_func,
        x0=x0,
        fprime=jacobian_func, 
        lb=lb,
        ub=ub
    )
    
    end_time = time.time()

    # --- 7. 结果展示与分析 ---
    duration_ms = (end_time - start_time) * 1000
    result_x = result["x"]

    print(f"\n[求解结果]")
    print(f"  Success: {result['success']}")
    if result["success"] and result["fun"] < 1e-5:
        print("真正意义上的求解成功")
    elif result["status"] > 0:
        print(f"求解器停滞在局部点，状态码: {result['status']}")
    else:
        print("求解失败")
    print(f"  最终解 x[0]: {result_x[0]:.6f}")
    
    # 手动计算残差
    real_res = residual_func(result_x)
    print(f"  报告残差 Norm: {result['fun']:.3e}")
    print(f"  手动验证真实残差 Norm: {np.linalg.norm(real_res):.3e}")

    # 芝诺现象判定
    dist_to_lb = np.abs(result_x[0] - lb[0])
    print(f"  距离值: {dist_to_lb:.3e}")
    if dist_to_lb < 1e-6 and not result['success']:
        print("\n>>> 结论：复现成功！")
        print("现象描述：解在 2.0，但牛顿法在 0.8 处受局部斜率诱导，疯狂撞向 0.5 的墙。")
        print("由于硬约束阻挡，它无法越过左侧的伪零点，最终导致收敛停滞。")

if __name__ == "__main__":
    main()