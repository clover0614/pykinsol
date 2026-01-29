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

namespace py = pybind11;
using namespace pybind11::literals;

// 常量定义
const int LINSOL_DENSE = 0;
const int LINSOL_GMRES = 1;

// 用户数据结构
struct KinsolUserData {
    py::function func;      // Python 残差函数 F(x)
    py::function jac_func;  // Python 雅可比函数 J(x)
    int N;                  // 维度
    bool has_jac;           // 是否提供雅可比
};

// --- N 维系统残差函数 ---
// 直接映射：N_Vector u (即 x) -> N_Vector f (即 F(x))
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    
    // 1. 将 u 包装为 numpy array (无拷贝视图)
    // 此时 u 就是 x，不再是松弛变量
    py::array_t<double> x_py = py::array_t<double>(
        {N}, {sizeof(double)}, NV_DATA_S(u), py::none()
    );

    // 2. 调用 Python 函数
    // 注意：Python 端的 func 内部已经包含了 x_safe = clip(x, lb, ub) 的逻辑
    // 所以这里直接传 x 进去即可
    py::array_t<double> res_py = data->func(x_py);
    
    // 3. 将结果填回 f
    auto res_proxy = res_py.unchecked<1>();
    double* f_data = NV_DATA_S(f);
    
    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i);
    }
    return 0;
}

// --- N 维雅可比函数 (仅 Dense 模式) ---
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    
    py::array_t<double> x_py = py::array_t<double>(
        {N}, {sizeof(double)}, NV_DATA_S(u), py::none()
    );

    py::array_t<double> jac_py = data->jac_func(x_py);
    auto jac_proxy = jac_py.unchecked<2>();
    double* J_data = SM_CONTENT_D(J)->data; 

    // 填充 NxN 矩阵 (列优先 Column-Major)
    // 这里不再有松弛变量的单位阵块，是非常纯粹的物理 Jacobian
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            J_data[i + j * N] = jac_proxy(i, j); 
        }
    }
    return 0;
}

// 核心实现
py::dict solve_cpp_impl(py::function func, py::array_t<double> x0, py::object jac, 
                        py::array_t<double> lb, py::array_t<double> ub,
                        int strategy, int linsol_type, int verbose) {
    
    int N = x0.size();
    SUNContext sunctx;
    SUNContext_Create(nullptr, &sunctx);

    KinsolUserData data;
    data.func = func;
    data.N = N;
    data.has_jac = !jac.is_none();
    if (data.has_jac) data.jac_func = jac.cast<py::function>();

    // 1. 初始化 u = x0
    // 直接使用原始维度 N
    N_Vector u = N_VNew_Serial(N, sunctx);
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) NV_Ith_S(u, i) = x0_p(i);

    // 2. 创建 KINSOL
    void* kin_mem = KINCreate(sunctx);
    KINInit(kin_mem, KinsolSysFn, u);
    KINSetUserData(kin_mem, &data);

    // 【重要】关于约束
    // 我们不在 C++ 层设置 KINSetConstraints。
    // 因为 KINSOL 原生只支持 x>0 这种简单约束，不支持任意 lb/ub。
    // 边界约束完全由 Python 端 func 中的 "Projection/Clipping" 逻辑处理。

    // 3. 设置日志级别
    KINSetPrintLevel(kin_mem, verbose);

    // 4. 配置线性求解器
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;

    if (linsol_type == LINSOL_DENSE) {
        J = SUNDenseMatrix(N, N, sunctx);
        LS = SUNLinSol_Dense(u, J, sunctx);
        KINSetLinearSolver(kin_mem, LS, J);
        if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn);
    } 
    else if (linsol_type == LINSOL_GMRES) {
        // GMRES + 差分 (Matrix-Free)
        LS = SUNLinSol_SPGMR(u, PREC_NONE, 0, sunctx);
        KINSetLinearSolver(kin_mem, LS, nullptr);
        // 如果想调整 GMRES 的参数（如重启维度），可以在这里调用 KINSpilsSetMaxl 等
    }

    // 缩放向量 (设为 1.0，依靠 Python 端的 Auto-Scaling)
    N_Vector scaling = N_VNew_Serial(N, sunctx);
    N_VConst(1.0, scaling); 
    
    // 5. 执行求解
    int flag = KINSol(kin_mem, u, strategy, scaling, scaling);
    
    N_VDestroy(scaling);

    // 6. 提取结果
    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = NV_Ith_S(u, i);
    
    // 计算最终残差范数
    py::array_t<double> res_py = data.func(x_res);
    auto res_p = res_py.unchecked<1>();
    double fnorm = 0.0;
    for (int i = 0; i < N; ++i) fnorm += res_p(i) * res_p(i);
    fnorm = std::sqrt(fnorm);

    // 清理
    N_VDestroy(u); 
    SUNLinSolFree(LS); 
    if (J) SUNMatDestroy(J); 
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    bool is_success = (flag >= 0); // KIN_SUCCESS(0), KIN_INITIAL_GUESS_OK(1), KIN_STEP_LT_STPTOL(2) 都算能用的结果

    return py::dict("x"_a=x_res, "fun"_a=fnorm, "success"_a=is_success, "status"_a=flag);
}

// 辅助转换
int get_strategy_enum(std::string method) {
    if (method == "newton" || method == "NEWTON") return KIN_NONE;
    return KIN_LINESEARCH;
}
int get_linsol_enum(std::string solver) {
    if (solver == "gmres" || solver == "GMRES") return LINSOL_GMRES;
    return LINSOL_DENSE;
}

// 包装接口 (保持签名与之前完全一致)
py::dict solve_cpp_wrapper(py::function func, py::array_t<double> x0, py::object jac, 
                           py::array_t<double> lb, py::array_t<double> ub,
                           std::string method, std::string linear_solver, 
                           int verbose) {
    
    // 虽然传入了 lb/ub，但在 C++ 内部我们只透传数据结构，不进行 KINSetConstraints 调用
    // 实际约束由 func 内部逻辑保证
    return solve_cpp_impl(func, x0, jac, lb, ub, 
                          get_strategy_enum(method), 
                          get_linsol_enum(linear_solver), 
                          verbose);
}

PYBIND11_MODULE(pykinsol, m) {
    m.doc() = "Kinsol wrapper (Direct N-dim)";

    m.def("pykinsol", &solve_cpp_wrapper, 
          "func"_a, 
          "x0"_a, 
          "fprime"_a = py::none(), 
          "lb"_a = py::none(), 
          "ub"_a = py::none(), 
          "method"_a = "linesearch", 
          "linear_solver"_a = "dense",
          "verbose"_a = 1
    );
    
    m.attr("KIN_NONE") = py::int_(KIN_NONE);
    m.attr("KIN_LINESEARCH") = py::int_(KIN_LINESEARCH);
}