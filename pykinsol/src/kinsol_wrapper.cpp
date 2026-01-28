#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h> 
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

// 定义内部常量用于线性求解器类型
const int LINSOL_DENSE = 0;
const int LINSOL_GMRES = 1;

// 用户数据结构体，用于向 SUNDIALS 传递 Python 回调和参数
struct KinsolUserData {
    py::function func;      // Python 残差函数
    py::function jac_func;  // Python 雅可比函数
    std::vector<double> lb; // 下界
    std::vector<double> ub; // 上界
    int N;                  // 维度
    bool has_jac;           // 是否提供了雅可比
};

// 辅助函数：将字符串策略转换为 KINSOL 内部整数常量
int get_strategy_enum(std::string method) {
    if (method == "newton" || method == "NEWTON") return KIN_NONE; // 纯牛顿法
    return KIN_LINESEARCH; // 默认使用线搜索
}

// 辅助函数：将字符串求解器类型转换为内部整数常量
int get_linsol_enum(std::string solver) {
    if (solver == "gmres" || solver == "GMRES") return LINSOL_GMRES; // GMRES 迭代求解
    return LINSOL_DENSE; // 默认使用稠密直接求解
}

// 系统残差函数 F(u) -> f
// 负责将松弛变量 u 映射回 x，计算 F(x)，并添加松弛约束方程
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u); // 输入变量 u (包含 s1, s2)
    double* f_data = NV_DATA_S(f); // 输出残差 f

    // 从松弛变量 s 恢复出原始变量 x
    // x = lb + s1 (s1 对应 u 的前 N 个元素)
    py::array_t<double> x_py(N);
    double* x_ptr = x_py.mutable_data();
    for (int i = 0; i < N; ++i) x_ptr[i] = data->lb[i] + s_data[i];

    // 调用 Python 残差函数计算物理残差 F(x)
    py::array_t<double> res_py = data->func(x_py);
    auto res_proxy = res_py.unchecked<1>();

    // 填充残差向量 f
    // f_i = F(x)_i  (物理方程)
    // f_{i+N} = s1_i + s2_i - (ub_i - lb_i)  (约束方程：确保 x 在 lb 和 ub 之间)
    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i);
        f_data[i + N] = s_data[i] + s_data[i + N] - (data->ub[i] - data->lb[i]);
    }
    return 0;
}

// 雅可比矩阵函数 J(u) (仅用于 Dense 模式)
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u);
    double* J_data = SM_CONTENT_D(J)->data; 
    int M = 2 * N; // 系统总维度

    // 恢复 x
    py::array_t<double> x_py(N);
    for (int i = 0; i < N; ++i) x_py.mutable_data()[i] = data->lb[i] + s_data[i];

    // 调用 Python 雅可比函数
    py::array_t<double> jac_py = data->jac_func(x_py);
    auto jac_proxy = jac_py.unchecked<2>();

    // 初始化矩阵为 0
    std::fill(J_data, J_data + M * M, 0.0);

    // 填充稠密雅可比矩阵 (列优先存储 Column-Major)
    for (int j = 0; j < N; ++j) {
        // 填充左上角：dF/dx
        for (int i = 0; i < N; ++i) {
            J_data[i + j * M] = jac_proxy(i, j);
        }
        // 填充约束部分的导数 (恒为 1.0)
        // ds1/ds1 = 1, ds2/ds2 = 1
        J_data[(j + N) + j * M] = 1.0; 
        J_data[(j + N) + (j + N) * M] = 1.0; 
    }
    return 0;
}

// 主求解器实现 (C++ 内部逻辑)
py::dict solve_cpp_impl(py::function func, py::array_t<double> x0, py::object jac, 
                        py::array_t<double> lb, py::array_t<double> ub,
                        int strategy, int linsol_type) {
    
    int N = x0.size();
    SUNContext sunctx;
    SUNContext_Create(nullptr, &sunctx); // 创建 SUNDIALS 上下文

    KinsolUserData data;
    data.func = func;
    data.N = N;
    data.lb.assign(lb.data(), lb.data() + N);
    data.ub.assign(ub.data(), ub.data() + N);
    data.has_jac = !jac.is_none();
    if (data.has_jac) data.jac_func = jac.cast<py::function>();

    // 初始化松弛变量 u
    // u = [s1, s2] 其中 s1 = x - lb, s2 = ub - x
    N_Vector u = N_VNew_Serial(2 * N, sunctx);
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) {
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i];
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i);
    }

    // 创建 KINSOL 内存并初始化
    void* kin_mem = KINCreate(sunctx);
    KINInit(kin_mem, KinsolSysFn, u);
    KINSetUserData(kin_mem, &data);

    // 设置硬约束：要求所有 s >= 0
    N_Vector constr = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, constr); 
    KINSetConstraints(kin_mem, constr);

    // 配置线性求解器
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;

    if (linsol_type == LINSOL_DENSE) {
        // 策略 A: 稠密直接求解 (适合小规模问题)
        J = SUNDenseMatrix(2 * N, 2 * N, sunctx);
        LS = SUNLinSol_Dense(u, J, sunctx);
        KINSetLinearSolver(kin_mem, LS, J);
        
        // 只有在 Dense 模式下才需要设置雅可比函数
        if (data.has_jac) {
            KINSetJacFn(kin_mem, KinsolJacFn);
        }
    } 
    else if (linsol_type == LINSOL_GMRES) {
        // 策略 B: GMRES 迭代求解 (适合大规模稀疏问题)
        // 使用 Matrix-Free 模式 (J 为 NULL)，内部使用差分法计算 J*v
        LS = SUNLinSol_SPGMR(u, PREC_NONE, 0, sunctx);
        KINSetLinearSolver(kin_mem, LS, nullptr);
    }

    // 设置缩放向量 (全为 1.0，表示不缩放)
    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); 
    
    // 执行求解
    // 如果需要调试，可以在此处调用 KINSetPrintLevel(kin_mem, 3);
    int flag = KINSol(kin_mem, u, strategy, scaling, scaling);
    
    N_VDestroy(scaling);

    bool is_success = (flag >= 0);

    // 从 u 中提取最终结果 x
    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    // 计算纯物理残差范数 (用于报告)
    // 过滤掉松弛变量引入的数值误差，只看 F(x)
    py::array_t<double> phys_res = data.func(x_res); 
    auto res_p = phys_res.unchecked<1>();
    double phys_fnorm = 0.0;
    for (int i = 0; i < N; ++i) {
        phys_fnorm += res_p(i) * res_p(i);
    }
    phys_fnorm = std::sqrt(phys_fnorm);

    // 释放资源
    N_VDestroy(u); N_VDestroy(constr); 
    SUNLinSolFree(LS); 
    if (J) SUNMatDestroy(J); 
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    // 返回结果字典
    return py::dict("x"_a=x_res, "fun"_a=phys_fnorm, "success"_a=is_success, "status"_a=flag);
}

// 包装函数：处理来自 Python 的字符串参数
py::dict solve_cpp_wrapper(py::function func, py::array_t<double> x0, py::object jac, 
                           py::array_t<double> lb, py::array_t<double> ub,
                           std::string method, std::string linear_solver) {
    
    // 将字符串参数转换为内部整数常量
    int strategy_int = get_strategy_enum(method);
    int linsol_int = get_linsol_enum(linear_solver);

    // 调用实现层
    return solve_cpp_impl(func, x0, jac, lb, ub, strategy_int, linsol_int);
}

// 模块定义
// 模块名必须与 setup.py 中的 "pykinsol" 保持一致
PYBIND11_MODULE(pykinsol, m) {
    m.doc() = "Kinsol solver top-level module (C++ backend)";

    // 导出主求解函数
    // 允许用户以 pykinsol.pykinsol(...) 的方式调用
    m.def("pykinsol", &solve_cpp_wrapper, 
          "Solve nonlinear system F(x)=0 with box constraints.",
          "func"_a, 
          "x0"_a, 
          "jac"_a = py::none(),      // 可选参数：雅可比
          "lb"_a = py::none(),       // 可选参数：下界
          "ub"_a = py::none(),       // 可选参数：上界
          "method"_a = "linesearch", // 默认参数：线搜索
          "linear_solver"_a = "dense" // 默认参数：稠密求解
    );
    
    // 导出常量 (可选，供用户检查)
    m.attr("KIN_NONE") = py::int_(KIN_NONE);
    m.attr("KIN_LINESEARCH") = py::int_(KIN_LINESEARCH);
}