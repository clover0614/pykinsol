import time
import numpy as np
from pykinsol import kinsol

def main():
    # --- 1. 问题规模与参数设置 ---
    N = 4
    print(f"=== 测试场景: {N} 维非线性方程组，各维度不同约束 ===")
    
    # --- 2. 定义目标真值 (Ground Truth) ---
    # 这里我们手动设置一个在约束范围内的解
    x_true = np.array([0.5, 1.2, 2.0, 0.8])
    
    # --- 3. 定义各维度不同的双边约束 ---
    # 第1维: [0.1, 2.0]
    # 第2维: [0.5, 3.0]
    # 第3维: [1.0, 4.0]
    # 第4维: [0.0, 1.5]
    lb = np.array([0.1, 0.5, 1.0, 0.0])
    ub = np.array([2.0, 3.0, 4.0, 1.5])
    
    # --- 4. 构造非线性方程组 ---
    # 方程组定义:
    # f1 = x1^2 + sin(x2) - 1.25 = 0
    # f2 = x1 * x2 - x3 + 0.5 = 0
    # f3 = exp(-x3) + x4^3 - 1.2 = 0
    # f4 = x1 + x2 + x3 + x4 - 4.5 = 0
    
    # 计算RHS使得x_true成为精确解
    x1, x2, x3, x4 = x_true
    
    f1 = x1**2 + np.sin(x2)
    f2 = x1 * x2 - x3
    f3 = np.exp(-x3) + x4**3
    f4 = x1 + x2 + x3 + x4
    
    # 设置目标值（方程右边）
    target = np.array([1.25, -0.5, 1.2, 4.5])
    
    # 计算RHS向量
    rhs_vector = np.array([f1, f2, f3, f4]) - target
    
    # --- 5. 定义残差函数 ---
    def residual_func(x):
        x1, x2, x3, x4 = x
        f = np.zeros(N)
        
        f[0] = x1**2 + np.sin(x2) - target[0]
        f[1] = x1 * x2 - x3 - target[1]
        f[2] = np.exp(-x3) + x4**3 - target[2]
        f[3] = x1 + x2 + x3 + x4 - target[3]
        
        return f - rhs_vector  # 减去RHS使得真值为零
    
    # --- 6. 定义雅可比矩阵 ---
    def jacobian_func(x):
        x1, x2, x3, x4 = x
        J = np.zeros((N, N))
        
        # 第一行: f1 = x1^2 + sin(x2) - 1.25
        J[0, 0] = 2 * x1
        J[0, 1] = np.cos(x2)
        J[0, 2] = 0
        J[0, 3] = 0
        
        # 第二行: f2 = x1 * x2 - x3 + 0.5
        J[1, 0] = x2
        J[1, 1] = x1
        J[1, 2] = -1
        J[1, 3] = 0
        
        # 第三行: f3 = exp(-x3) + x4^3 - 1.2
        J[2, 0] = 0
        J[2, 1] = 0
        J[2, 2] = -np.exp(-x3)
        J[2, 3] = 3 * x4**2
        
        # 第四行: f4 = x1 + x2 + x3 + x4 - 4.5
        J[3, 0] = 1
        J[3, 1] = 1
        J[3, 2] = 1
        J[3, 3] = 1
        
        return J
    
    # --- 7. 初始检查 ---
    # 初始猜想值设置为约束范围的中点
    x0 = (lb + ub) / 2
    
    print("-" * 50)
    print("=== 状态预检 ===")
    print(f"各维度约束下限 lb: {lb}")
    print(f"各维度约束上限 ub: {ub}")

    print(f"初始猜想值 x0: {x0}")
    init_res = residual_func(x0)
    print(f"初始物理残差 Norm: {np.linalg.norm(init_res):.6e}")

    print(f"已知可行解 x_true: {x_true}")
    feasible = np.all((x_true >= lb - 1e-9) & (x_true <= ub + 1e-9))
    print(f"真值是否满足约束: {feasible}")
    print(f"真值残差验证: {np.linalg.norm(residual_func(x_true)):.6e}")
    print("-" * 50)
    
    # --- 8. 调用求解器 ---
    print(f"\n=== 开始求解 (维度: {N}) ===")
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
        print(f"求解耗时: {duration_ms:.3f} ms")
        print(f"最终解: {result_x}")
        print(f"最终残差 Norm: {result['fun']:.3e}")
        
        # 验证约束
        within_bounds = np.all((result_x >= lb - 1e-9) & (result_x <= ub + 1e-9))
        print(f"约束满足检查: {within_bounds}")
        
        # 计算真实残差
        manual_res_vector = residual_func(result_x)
        manual_norm = np.linalg.norm(manual_res_vector)
        print(f"手动计算残差 Norm: {manual_norm:.3e}")
        
        # 与真值比较
        diff = np.linalg.norm(result_x - x_true)
        print(f"与已知真值的差异: {diff:.3e}")
        
        # 显示各方程的具体残差
        print("\n各方程具体残差:")
        for i in range(N):
            print(f"  方程 {i+1}: {manual_res_vector[i]:.3e}")
            
    else:
        print(f"求解失败。状态码: {result.get('status')}, 最终残差: {result['fun']:.3e}")
    
    print("="*50)

if __name__ == "__main__":
    main()