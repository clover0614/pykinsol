#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h> // [新增] GMRES 线性求解器
#include <vector>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

// 定义常量供 Python 调用
const int LINSOL_DENSE = 0;
const int LINSOL_GMRES = 1;

struct KinsolUserData {
    py::function func;
    py::function jac_func;
    std::vector<double> lb;
    std::vector<double> ub;
    int N;
    bool has_jac;
};

// 系统残差函数 (保持不变)
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u);
    double* f_data = NV_DATA_S(f);

    py::array_t<double> x_py(N);
    double* x_ptr = x_py.mutable_data();
    for (int i = 0; i < N; ++i) x_ptr[i] = data->lb[i] + s_data[i];

    py::array_t<double> res_py = data->func(x_py);
    auto res_proxy = res_py.unchecked<1>();

    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i);
        f_data[i + N] = s_data[i] + s_data[i + N] - (data->ub[i] - data->lb[i]);
    }
    return 0;
}

// 雅可比矩阵函数 (仅用于 Dense 模式)
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u);
    double* J_data = SM_CONTENT_D(J)->data; 
    int M = 2 * N;

    py::array_t<double> x_py(N);
    for (int i = 0; i < N; ++i) x_py.mutable_data()[i] = data->lb[i] + s_data[i];

    py::array_t<double> jac_py = data->jac_func(x_py);
    auto jac_proxy = jac_py.unchecked<2>();

    std::fill(J_data, J_data + M * M, 0.0);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            J_data[i + j * M] = jac_proxy(i, j);
        }
        J_data[(j + N) + j * M] = 1.0; 
        J_data[(j + N) + (j + N) * M] = 1.0; 
    }
    return 0;
}

// 主求解函数
// [修改] 增加 strategy 和 linsol_type 参数
py::dict solve_cpp(py::function func, py::array_t<double> x0, py::object jac, 
                   py::array_t<double> lb, py::array_t<double> ub,
                   int strategy, int linsol_type) {
    
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

    N_Vector u = N_VNew_Serial(2 * N, sunctx);
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) {
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i];
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i);
    }

    void* kin_mem = KINCreate(sunctx);
    KINInit(kin_mem, KinsolSysFn, u);
    KINSetUserData(kin_mem, &data);

    N_Vector constr = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, constr); 
    KINSetConstraints(kin_mem, constr);

    // ==========================================
    // [修改] 线性求解器选择逻辑
    // ==========================================
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;

    if (linsol_type == LINSOL_DENSE) {
        // --- 策略 A: 稠密矩阵直接求解 (适合小规模 < 5000) ---
        J = SUNDenseMatrix(2 * N, 2 * N, sunctx);
        LS = SUNLinSol_Dense(u, J, sunctx);
        KINSetLinearSolver(kin_mem, LS, J);
        
        // 只有在 Dense 模式下，我们才显式使用 Python 提供的 Jacobian 矩阵
        if (data.has_jac) {
            KINSetJacFn(kin_mem, KinsolJacFn);
        }
    } 
    else if (linsol_type == LINSOL_GMRES) {
        // --- 策略 B: GMRES 迭代求解 (适合大规模稀疏) ---
        // 使用 Matrix-Free 模式 (J 为 NULL)
        // KINSOL 默认会使用差分法计算 J*v，这对于大系统很高效
        LS = SUNLinSol_SPGMR(u, PREC_NONE, 0, sunctx);
        KINSetLinearSolver(kin_mem, LS, nullptr);
        
        // 注意：GMRES 模式下，如果用户提供了 jac，我们需要实现 JacTimesVecFn
        // 简单起见，这里暂不使用 Python 的 jac，而是依赖 KINSOL 的内部差分
        // 如果需要更高性能，后续可以封装 KINSetJacTimesVecFn
    }

    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); 
    
    // ==========================================
    // [修改] 使用传入的 strategy 参数 (KIN_NONE 或 KIN_LINESEARCH)
    // ==========================================
    int flag = KINSol(kin_mem, u, strategy, scaling, scaling);
    
    N_VDestroy(scaling);

    bool is_success = (flag >= 0);

    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    // 手动计算物理残差
    py::array_t<double> phys_res = data.func(x_res); 
    auto res_p = phys_res.unchecked<1>();
    double phys_fnorm = 0.0;
    for (int i = 0; i < N; ++i) {
        phys_fnorm += res_p(i) * res_p(i);
    }
    phys_fnorm = std::sqrt(phys_fnorm);

    // 资源清理
    N_VDestroy(u); N_VDestroy(constr); 
    SUNLinSolFree(LS); 
    if (J) SUNMatDestroy(J); // 只有 Dense 模式下 J 才不为空
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    return py::dict("x"_a=x_res, "fun"_a=phys_fnorm, "success"_a=is_success, "status"_a=flag);
}

PYBIND11_MODULE(kinsol_cpp, m) {
    // 导出常量供 Python 使用
    m.attr("KIN_NONE") = py::int_(KIN_NONE);
    m.attr("KIN_LINESEARCH") = py::int_(KIN_LINESEARCH);
    m.attr("LINSOL_DENSE") = py::int_(LINSOL_DENSE);
    m.attr("LINSOL_GMRES") = py::int_(LINSOL_GMRES);

    m.def("solve", &solve_cpp, 
          "Solve nonlinear system",
          "func"_a, "x0"_a, "jac"_a, "lb"_a, "ub"_a, 
          "strategy"_a, "linsol_type"_a); // 注册新参数
}