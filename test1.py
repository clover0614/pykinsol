import numpy as np
import sys
import os
import pykinsol

print(f"当前使用的包路径: {os.path.dirname(pykinsol.__file__)}")

# 确保能找到包
sys.path.append(os.path.dirname(__file__))
from pykinsol import kinsol

def solve_case(case_name, lb_val, ub_val, expect_success):
    print(f"\n{'='*20} 测试场景: {case_name} {'='*20}")
    
    # --- 1. 定义方程 (2维) ---
    # F[0] = x^2 + y^2 - 1
    # F[1] = x - y
    def func(x):
        return np.array([
            x[0]**2 + x[1]**2 - 1.0,
            x[0] - x[1]
        ])

    # --- 2. 定义雅可比 (2x2) ---
    def jac(x):
        # J = [[2x, 2y], 
        #      [1,  -1]]
        return np.array([
            [2*x[0], 2*x[1]],
            [1.0,    -1.0]
        ])

    # --- 3. 设置约束 ---
    lb = np.array([lb_val, lb_val])
    ub = np.array([ub_val, ub_val])
    x0 = np.array([lb_val, lb_val]) # 从边界起步

    # 如果边界在真解(0.707)之外，我们需要一个稍微靠谱点的初值方向
    # 比如设置初值等于下界
    
    print(f"约束范围: [{lb[0]}, {ub[0]}]")
    print(f"理论真解: (0.7071, 0.7071)")
    if not expect_success:
        print(">>> 预期结果: 求解失败 (Success=False) 或 残差巨大")
    else:
        print(">>> 预期结果: 求解成功 (Success=True) 且 残差接近 0")

    # --- 4. 求解 ---
    res = kinsol(func, x0, jac, lb, ub)
    
    # --- 5. 核心诊断逻辑 ---
    print(f"\n[求解器返回结果]")
    print(f"  Success: {res['success']}")
    print(f"  Status:  {res.get('status')}")
    print(f"  Message: {res.get('message', 'N/A')}")
    print(f"  解 x:    {res['x']}")
    print(f"  报告残差 (res['fun']): {res['fun']:.6e}")
    
    # --- 6. 手动照妖镜 ---
    real_f = func(res['x'])
    real_norm = np.linalg.norm(real_f)
    print(f"[手动计算验证]")
    print(f"  真实残差 Norm: {real_norm:.6e}")
    
    # 判定
    if expect_success:
        if res['success'] and real_norm < 1e-5:
            print("✅ [正常组] 测试通过：求解器工作正常。")
        else:
            print("❌ [正常组] 测试失败：本该解出来却没解出来。")
    else:
        # 陷阱组：真解(0.7)在约束(2.0)之外
        # 在 (2,2) 处，F[0] = 4+4-1=7, F[1]=0. Norm=7.0
        if res['success']:
            if real_norm > 1.0:
                print("🔴 [严重 BUG 确认]：求解器报告成功，但实际残差巨大！")
                print("   原因推测：封装层可能只检查了松弛变量的残差(0)，或者忽略了错误码。")
            else:
                print("❓ [奇怪]：求解器报告成功，且残差很小？这意味着它突破了你的约束限制！检查 res['x'] 是否小于 lb。")
        else:
            print("✅ [陷阱组] 测试通过：求解器正确报告了失败（或者我们捕获到了异常）。")

def main():
    # 用例 1: 正常求解
    # 约束 [-2, 2]，真解 0.707 在范围内
    solve_case("Case 1: 正常范围", -2.0, 2.0, expect_success=True)

    # 用例 2: 制造“芝诺陷阱”
    # 约束 [2.0, 5.0]，真解 0.707 在范围外
    # 求解器应该卡在 (2.0, 2.0)，残差应该约为 7.0
    solve_case("Case 2: 范围外死锁", 2.0, 5.0, expect_success=False)

if __name__ == "__main__":
    main()