#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <vector>

namespace py = pybind11; // 简化命名空间
using namespace pybind11::literals;

struct KinsolUserData { // 用户数据结构体
    py::function func;
    py::function jac_func;
    std::vector<double> lb;
    std::vector<double> ub;
    int N;
    bool has_jac;
};

// 系统残差函数
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u); // 获取输入变量 u 的数据指针 (长度 2N)
    double* f_data = NV_DATA_S(f); // 获取输出残差 f 的数据指针 (长度 2N)

    // 从松弛变量 s1 恢复出 x
    py::array_t<double> x_py(N);
    double* x_ptr = x_py.mutable_data();
    for (int i = 0; i < N; ++i) x_ptr[i] = data->lb[i] + s_data[i];

    // 调用 Python 函数计算原方程残差
    py::array_t<double> res_py = data->func(x_py);
    auto res_proxy = res_py.unchecked<1>(); // 获取快速访问代理

    // 填充 SUNDIALS 残差向量 f
    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i);
        f_data[i + N] = s_data[i] + s_data[i + N] - (data->ub[i] - data->lb[i]);
    }
    return 0;
}

// 雅可比矩阵函数
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u);
    double* J_data = SM_CONTENT_D(J)->data; // 获取稠密矩阵的数据指针
    int M = 2 * N; // 矩阵维数

    // 恢复x
    py::array_t<double> x_py(N);
    for (int i = 0; i < N; ++i) x_py.mutable_data()[i] = data->lb[i] + s_data[i];

    // 调用 Python 计算原问题雅可比 J(x)
    py::array_t<double> jac_py = data->jac_func(x_py);
    auto jac_proxy = jac_py.unchecked<2>();

    // 初始化矩阵为 0
    std::fill(J_data, J_data + M * M, 0.0);

    // 填充矩阵，SUNDIALS是列优先矩阵 index = row + col * M
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            J_data[i + j * M] = jac_proxy(i, j);
        }
        J_data[(j + N) + j * M] = 1.0; // 左下角
        J_data[(j + N) + (j + N) * M] = 1.0; // 右下角
    }
    return 0;
}

// 主求解函数
py::dict solve_cpp(py::function func, py::array_t<double> x0, py::object jac, 
                   py::array_t<double> lb, py::array_t<double> ub) {
    int N = x0.size();
    SUNContext sunctx;
    SUNContext_Create(nullptr, &sunctx); // 创建上下文

    KinsolUserData data;
    data.func = func;
    data.N = N;
    data.lb.assign(lb.data(), lb.data() + N);
    data.ub.assign(ub.data(), ub.data() + N);
    data.has_jac = !jac.is_none();
    if (data.has_jac) data.jac_func = jac.cast<py::function>();

    N_Vector u = N_VNew_Serial(2 * N, sunctx); // 创建 2N 长度的向量
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) {
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i]; // s1 = x - lb
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i); // s2 = ub - x
    }

    // 配置KINSOL 
    void* kin_mem = KINCreate(sunctx); // 创建求解器内存
    KINInit(kin_mem, KinsolSysFn, u); // 初始化，绑定系统方程
    KINSetUserData(kin_mem, &data);

    N_Vector constr = N_VNew_Serial(2 * N, sunctx);
    // 使用KINSOL的原生约束 s>=0，这里应该设为 1.0
    N_VConst(1.0, constr); 
    KINSetConstraints(kin_mem, constr);

    // 设置线性求解器
    SUNMatrix J = SUNDenseMatrix(2 * N, 2 * N, sunctx); // 创建稠密矩阵
    SUNLinearSolver LS = SUNLinSol_Dense(u, J, sunctx); // 创建稠密线性求解器
    KINSetLinearSolver(kin_mem, LS, J); // 绑定到 KINSOL
    if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn);

    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); // 设为 1.0，表示不缩放
    // 使用 scaling 作为缩放参数，u_scale (状态变量缩放) 和 f_scale (函数值缩放)
    int flag = KINSol(kin_mem, u, KIN_LINESEARCH, scaling, scaling);
    N_VDestroy(scaling); // 释放 scaling

    bool is_success = (flag >= 0);

    // 将结果从 s1 转换回 x
    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    // 获取最终的残差范数 (Residual Norm)
    double fnorm;
    KINGetFuncNorm(kin_mem, &fnorm);

    N_VDestroy(u); N_VDestroy(constr); SUNLinSolFree(LS); SUNMatDestroy(J);
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    // 返回python字典
    return py::dict("x"_a=x_res, "fun"_a=fnorm, "success"_a=is_success, "status"_a=flag);
}

PYBIND11_MODULE(kinsol_cpp, m) {
    m.def("solve", &solve_cpp);
}