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
#include <iomanip>

namespace py = pybind11;
using namespace pybind11::literals;

// 定义内部常量用于线性求解器类型
const int LINSOL_DENSE = 0;
const int LINSOL_GMRES = 1;

// 用户数据结构体
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
    if (method == "newton" || method == "NEWTON") return KIN_NONE; 
    return KIN_LINESEARCH; // 默认使用线搜索
}

// 辅助函数：将字符串求解器类型转换为内部整数常量
int get_linsol_enum(std::string solver) {
    if (solver == "gmres" || solver == "GMRES") return LINSOL_GMRES; 
    return LINSOL_DENSE; 
}

// 系统残差函数 F(u) -> f
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u); // 输入变量 u (包含 s1, s2)
    double* f_data = NV_DATA_S(f); // 输出残差 f

    // 从松弛变量 s 恢复出原始变量 x
    py::array_t<double> x_py(N);
    double* x_ptr = x_py.mutable_data();
    for (int i = 0; i < N; ++i) x_ptr[i] = data->lb[i] + s_data[i];

    // 调用 Python 残差函数计算物理残差 F(x)
    py::array_t<double> res_py = data->func(x_py);
    auto res_proxy = res_py.unchecked<1>();

    // 填充残差向量 f
    // f_i = F(x)_i 
    // f_{i+N} = s1_i + s2_i - (ub_i - lb_i)
    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i);
        f_data[i + N] = s_data[i] + s_data[i + N] - (data->ub[i] - data->lb[i]);
    }

    // [已删除] 原先繁杂的 Debug 打印逻辑
    
    return 0;
}

// 雅可比矩阵函数 J(u) (仅用于 Dense 模式)
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u);
    double* J_data = SM_CONTENT_D(J)->data; 
    int M = 2 * N; 

    // 恢复 x
    py::array_t<double> x_py(N);
    for (int i = 0; i < N; ++i) x_py.mutable_data()[i] = data->lb[i] + s_data[i];

    // 调用 Python 雅可比函数
    py::array_t<double> jac_py = data->jac_func(x_py);
    auto jac_proxy = jac_py.unchecked<2>();

    // 初始化矩阵为 0
    std::fill(J_data, J_data + M * M, 0.0);

    // 填充稠密雅可比矩阵
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            J_data[i + j * M] = jac_proxy(i, j);
        }
        J_data[(j + N) + j * M] = 1.0; 
        J_data[(j + N) + (j + N) * M] = 1.0; 
    }
    return 0;
}

// 主求解器实现
py::dict solve_cpp_impl(py::function func, py::array_t<double> x0, py::object jac, 
                        py::array_t<double> lb, py::array_t<double> ub,
                        py::array_t<int> constraint_mask,
                        int strategy, int linsol_type, 
                        int verbose,
                        double fnormtol, double scsteptol,
                        py::dict options // <--- [新增] 接收 options 字典
                        ) { 
    
    int N = x0.size();
    SUNContext sunctx;
    SUNContext_Create(nullptr, &sunctx);

    KinsolUserData data;
    data.func = func;
    data.N = N;
    data.lb.assign(lb.data(), lb.data() + N);
    data.ub.assign(ub.data(), ub.data() + N);
    data.has_jac = !jac.is_none();
    if (data.has_jac) data.jac_func = jac.cast<py::function>();

    // 初始化松弛变量 u
    N_Vector u = N_VNew_Serial(2 * N, sunctx);
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) {
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i];      // s1
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i);  // s2
    }

    // 创建 KINSOL 内存
    void* kin_mem = KINCreate(sunctx);
    KINInit(kin_mem, KinsolSysFn, u);
    
    // 设置基础容差
    KINSetFuncNormTol(kin_mem, fnormtol);
    KINSetScaledStepTol(kin_mem, scsteptol);
    KINSetUserData(kin_mem, &data);

    // 设置约束向量
    N_Vector constr = N_VNew_Serial(2 * N, sunctx);
    double* constr_ptr = NV_DATA_S(constr);
    auto mask = constraint_mask.unchecked<1>();

    for (int i = 0; i < N; ++i) {
        if (mask(i) == 1) { 
            constr_ptr[i] = 2.0;     // s1 > 0
            constr_ptr[i + N] = 2.0; // s2 > 0
        } else {            
            constr_ptr[i] = 0.0;     
            constr_ptr[i + N] = 0.0; 
        }
    }
    KINSetConstraints(kin_mem, constr);

    // 设置日志打印级别 (verbose)
    KINSetPrintLevel(kin_mem, verbose);

    // --- [新增] 解析并设置 max_iter (非线性迭代次数) ---
    if (options.contains("max_iter")) {
        long int mxiter = options["max_iter"].cast<long int>();
        KINSetNumMaxIters(kin_mem, mxiter);
        if (verbose > 0) std::cout << "Info: Set max_iter = " << mxiter << std::endl;
    }

    // 配置线性求解器
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;

    if (linsol_type == LINSOL_DENSE) {
        J = SUNDenseMatrix(2 * N, 2 * N, sunctx);
        LS = SUNLinSol_Dense(u, J, sunctx);
        KINSetLinearSolver(kin_mem, LS, J);
        if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn);
    } 
    else if (linsol_type == LINSOL_GMRES) {
        // --- [新增] 解析 linear_iter 并传递给 SPGMR ---
        // 在 SUNDIALS 中，linear_iter 对应 maxl (Krylov 子空间重启维度)
        int maxl = 0; // 0 表示使用 SUNDIALS 默认值 (通常是 5)
        if (options.contains("linear_iter")) {
            maxl = options["linear_iter"].cast<int>();
            if (verbose > 0) std::cout << "Info: Set linear_iter (GMRES maxl) = " << maxl << std::endl;
        }

        LS = SUNLinSol_SPGMR(u, PREC_NONE, maxl, sunctx);
        KINSetLinearSolver(kin_mem, LS, nullptr);
    }

    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); 
    
    // [已删除] 原先的 x0 Consistency Check Debug 代码

    // 执行求解
    int flag = KINSol(kin_mem, u, strategy, scaling, scaling);
    
    N_VDestroy(scaling);

    bool is_success = (flag >= 0);

    // 提取结果 (从 s1 恢复 x)
    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    // 计算物理残差
    py::array_t<double> phys_res = data.func(x_res); 
    auto res_p = phys_res.unchecked<1>();
    double phys_fnorm = 0.0;
    for (int i = 0; i < N; ++i) phys_fnorm += res_p(i) * res_p(i);
    phys_fnorm = std::sqrt(phys_fnorm);

    // 清理资源
    N_VDestroy(u); N_VDestroy(constr); 
    SUNLinSolFree(LS); 
    if (J) SUNMatDestroy(J); 
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    return py::dict("x"_a=x_res, "fun"_a=phys_fnorm, "success"_a=is_success, "status"_a=flag);
}

// 包装函数
py::dict solve_cpp_wrapper(py::function func, py::array_t<double> x0, py::object jac, 
                           py::array_t<double> lb, py::array_t<double> ub,
                           py::array_t<int> constraint_mask,
                           std::string method, std::string linear_solver, 
                           int verbose, 
                           double fnormtol, double scsteptol,
                           py::dict options // <--- [新增]
                           ) {
    
    int strategy_int = get_strategy_enum(method);
    int linsol_int = get_linsol_enum(linear_solver);

    return solve_cpp_impl(func, x0, jac, lb, ub, constraint_mask, strategy_int, linsol_int, verbose, fnormtol, scsteptol, options);
}

// 模块定义
PYBIND11_MODULE(pykinsol, m) {
    m.doc() = "Kinsol solver with logging control and options";
    m.def("pykinsol", &solve_cpp_wrapper, 
          "func"_a, 
          "x0"_a, 
          "fprime"_a = py::none(), 
          "lb"_a = py::none(), 
          "ub"_a = py::none(), 
          "constraint_mask"_a,
          "method"_a = "linesearch", 
          "linear_solver"_a = "dense",
          "verbose"_a = 1, 
          "fnormtol"_a = 1e-8,   
          "scsteptol"_a = 1e-20,
          "options"_a = py::dict() // <--- [新增] 默认为空字典
    );
    
    m.attr("KIN_NONE") = py::int_(KIN_NONE);
    m.attr("KIN_LINESEARCH") = py::int_(KIN_LINESEARCH);
}