#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <vector>
#include <iostream> // 虽然去掉了调试打印，保留它是个好习惯

namespace py = pybind11;
using namespace pybind11::literals;

struct KinsolUserData {
    py::function func;
    py::function jac_func;
    std::vector<double> lb;
    std::vector<double> ub;
    int N;
    bool has_jac;
};

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

py::dict solve_cpp(py::function func, py::array_t<double> x0, py::object jac, 
                   py::array_t<double> lb, py::array_t<double> ub) {
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
    N_VConst(0.0, constr); // 注意：这里通常设为0表示无约束，但在你的逻辑里，是依靠松弛变量本身来保证的吗？
    // 如果你要使用KINSOL的原生约束 s>=0，这里应该设为 1.0
    // 根据之前的沟通，我们设为 1.0 以启用 s >= 0 约束
    N_VConst(1.0, constr); 
    KINSetConstraints(kin_mem, constr);

    SUNMatrix J = SUNDenseMatrix(2 * N, 2 * N, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(u, J, sunctx);
    KINSetLinearSolver(kin_mem, LS, J);
    if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn);

    // ==========================================
    // 【关键修复】创建 Scaling 向量并设为 1.0
    // ==========================================
    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); // 设为 1.0，表示不缩放

    // 使用 scaling 作为缩放参数，而不是 u
    int flag = KINSol(kin_mem, u, KIN_LINESEARCH, scaling, scaling);
    
    // 释放 scaling
    N_VDestroy(scaling);
    // ==========================================

    bool is_success = (flag >= 0);

    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    double fnorm;
    KINGetFuncNorm(kin_mem, &fnorm);

    N_VDestroy(u); N_VDestroy(constr); SUNLinSolFree(LS); SUNMatDestroy(J);
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    return py::dict("x"_a=x_res, "fun"_a=fnorm, "success"_a=is_success, "status"_a=flag);
}

PYBIND11_MODULE(kinsol_cpp, m) {
    m.def("solve", &solve_cpp);
}