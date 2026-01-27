import time
import numpy as np
import sys
import os
from pykinsol import kinsol

def main():
    # --- 1. 问题规模设置 (2D Bratu Problem) ---
    N = 60
    size = N * N
    lam = 6.0
    h = 1.0 / (N - 1)  # 网格间距，是一个常数
    h_sq = h * h
    
    print(f"=== 测试场景: 2D Bratu Problem ===")
    print(f"网格规模: {N}x{N}, 变量总数: {size}")

    # --- 2. 定义约束边界 ---
    lb = np.full(size, 0.0)
    ub = np.full(size, 6.0)

    # --- 3. 定义 2D Bratu 残差函数 ---
    def bratu_residual(u_vec):
        u = u_vec.reshape((N, N))
        f = np.zeros((N, N))

        center = u[1:-1, 1:-1]
        up     = u[0:-2, 1:-1]
        down   = u[2:,   1:-1]
        left   = u[1:-1, 0:-2]
        right  = u[1:-1, 2:]

        laplacian = (up + down + left + right - 4.0 * center) / h_sq
        nonlinear = lam * np.exp(center)

        f[1:-1, 1:-1] = laplacian + nonlinear

        # Dirichlet 边界
        f[0, :]  = u[0, :]
        f[-1, :] = u[-1, :]
        f[:, 0]  = u[:, 0]
        f[:, -1] = u[:, -1]

        return f.flatten()

    # --- 4. 定义 2D Bratu 雅可比矩阵 ---
    def bratu_jacobian(u_vec):
        jac = np.zeros((size, size))
        u = u_vec.reshape((N, N))
        
        # 填充雅可比矩阵
        for i in range(N):
            for j in range(N):
                row = i * N + j
                # 边界点：df_i/du_i = 1
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    jac[row, row] = 1.0
                else:
                    # 内部点：拉普拉斯部分 + 非线性导数部分
                    # 对角项: -4/h^2 + lambda * exp(u_i)
                    jac[row, row] = -4.0 / h_sq + lam * np.exp(u[i, j])
                    # 相邻项 (上下左右): 1/h^2
                    jac[row, row - N] = 1.0 / h_sq # 上
                    jac[row, row + N] = 1.0 / h_sq # 下
                    jac[row, row - 1] = 1.0 / h_sq # 左
                    jac[row, row + 1] = 1.0 / h_sq # 右
        return jac

    # --- 5. 初始猜想 ---
    x0 = np.full(size, 0.1)
    print("-" * 30)
    print("=== 状态预检 ===")
    init_res = bratu_residual(x0)
    print(f"初始物理残差 Norm: {np.linalg.norm(init_res):.6e}")
    print("-" * 30)

    # --- 6. 开始求解 ---
    print(f"=== 开始求解 (内部维度: {2*size}) ===")
    start_time = time.time()
    
    result = kinsol(
        func=bratu_residual,
        x0=x0,
        fprime=bratu_jacobian,
        lb=lb,
        ub=ub
    )
    
    end_time = time.time()

    # --- 7. 输出与分析 ---
    if result["success"]:
        result_x = result["x"]
        duration_ms = (end_time - start_time) * 1000
        
        print(f"求解成功!")
        print(f"总耗时: {duration_ms:.3f} ms")
        print(f"最终解 x: {result_x}")
        print(f"最终残差范数: {result['fun']:.3e}")
        print(f"解统计: Min={np.min(result_x):.4f}, Max={np.max(result_x):.4f}")
        
        manual_res_vector = bratu_residual(result_x)
        manual_norm = np.linalg.norm(manual_res_vector)
        print(f"手动计算验证: {manual_norm:.3e}")

        # 边界约束检查
        in_bounds = np.all((result_x >= lb - 1e-7) & (result_x <= ub + 1e-7))
        print(f"约束满足检查: {in_bounds}")
        
        # 验证中心点
        center_val = result_x.reshape((N, N))[N//2, N//2]
        print(f"网格中心点值: {center_val:.6f}")
    else:
        print(f"求解失败。状态码: {result.get('status')}, 最终残差: {result['fun']:.3e}")

if __name__ == "__main__":
    main()