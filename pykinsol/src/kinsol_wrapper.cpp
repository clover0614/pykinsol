//kinsol_wrapper 实现Python封装与松弛变量变换
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <vector>

namespace py = pybind11; //定义别名
using namespace pybind11::literals; // 必须引入此命名空间才能使用 "_a"

// 数据交换结构体，用于在 C++ 回调函数和 Python 之间传递数据
struct KinsolUserData {
    py::function func;
    py::function jac_func;
    std::vector<double> lb;
    std::vector<double> ub;
    int N;
    bool has_jac;
};

// 系统残差函数回调
static int KinsolSysFn(N_Vector u, N_Vector f, void *user_data) {
    // 取出打包好的 Python 对象，从void*转换为KinsolUserData*类型
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    // 使用SUNDIALS的宏NV_DATA_S从N_Vector中提取原始数据指针
    double* s_data = NV_DATA_S(u); // 松弛变量 [2N]
    double* f_data = NV_DATA_S(f); // 残差函数

    // 1) 把 s_x 还原成 x
    py::array_t<double> x_py(N); // 创建N维py数组
    double* x_ptr = x_py.mutable_data(); // 获取可写数据指针
    for (int i = 0; i < N; ++i) x_ptr[i] = data->lb[i] + s_data[i]; // 将松弛变量转换回原始变量 x=lb+s1

    py::array_t<double> res_py = data->func(x_py); // 计算原始问题的残差 F(x)
    auto res_proxy = res_py.unchecked<1>(); // 创建res_py的代理对象，<1>一维数组

    for (int i = 0; i < N; ++i) {
        f_data[i] = res_proxy(i); // 用户给的N个方程
        f_data[i + N] = s_data[i] + s_data[i + N] - (data->ub[i] - data->lb[i]); // N个一致性方程，s1+s2-lb-ub=0
    }
    return 0;
}

// 雅可比矩阵回调，计算残差对变量的导数
static int KinsolJacFn(N_Vector u, N_Vector fu, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    auto* data = static_cast<KinsolUserData*>(user_data);
    int N = data->N;
    double* s_data = NV_DATA_S(u); // 使用SUNDIALS宏NV_DATA_S从N_Vector中提取原始数据指针
    double* J_data = SM_CONTENT_D(J)->data;  // 使用SM_CONTENT_D宏访问SUNMatrix的底层数据结构
    int M = 2 * N;

    py::array_t<double> x_py(N); // 使用pybind11创建N维double类型Python数组
    for (int i = 0; i < N; ++i) x_py.mutable_data()[i] = data->lb[i] + s_data[i]; // x = lb + s1

    py::array_t<double> jac_py = data->jac_func(x_py); // 原始雅可比
    auto jac_proxy = jac_py.unchecked<2>(); // 使用unchecked方法创建二维数组代理

    std::fill(J_data, J_data + M * M, 0.0); // 初始化雅可比矩阵

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            J_data[i + j * M] = jac_proxy(i, j); // 左上，dF1/ds1
        }
        J_data[(j + N) + j * M] = 1.0; // 左下，dF2/ds1
        J_data[(j + N) + (j + N) * M] = 1.0; // 右下，dF2/ds2
    }
    return 0;
}

// 求解主函数，暴露给python，返回Python字典
py::dict solve_cpp(py::function func, py::array_t<double> x0, py::object jac, 
                  py::array_t<double> lb, py::array_t<double> ub) {
    int N = x0.size();
    SUNContext sunctx; // 创建 SUNDIALS 上下文
    SUNContext_Create(nullptr, &sunctx); // 初始化上下文，管理库资源

    KinsolUserData data; // 初始化用户数据结构
    data.func = func;
    data.N = N;
    data.lb.assign(lb.data(), lb.data() + N); // 复制到data.lb中
    data.ub.assign(ub.data(), ub.data() + N);
    data.has_jac = !jac.is_none(); // 检查是否提供了雅可比函数
    if (data.has_jac) data.jac_func = jac.cast<py::function>(); // 转换为 py::function

    N_Vector u = N_VNew_Serial(2 * N, sunctx); // 创建新的序列向量
    auto x0_p = x0.unchecked<1>(); // 创建初值x0的代理
    for (int i = 0; i < N; ++i) {
        //  NV_Ith_S 是 SUNDIALS 宏，访问向量的第 i 个元素
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i]; // s1 = x - lb
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i); // s2 = ub - x
    }

    void* kin_mem = KINCreate(sunctx); // 创建 KINSOL 求解器内存块，返回指针
    KINInit(kin_mem, KinsolSysFn, u); // 初始化求解器
    KINSetUserData(kin_mem, &data); // 设置用户数据，传入data结构体

    N_Vector constr = N_VNew_Serial(2 * N, sunctx); // 创建2N维的约束向量
    N_VConst(1.0, constr); // 初始化为1，表示无约束
    KINSetConstraints(kin_mem, constr);

    SUNMatrix J = SUNDenseMatrix(2 * N, 2 * N, sunctx); // 创建稠密矩阵
    SUNLinearSolver LS = SUNLinSol_Dense(u, J, sunctx); // 创建稠密矩阵的直接线性求解器
    KINSetLinearSolver(kin_mem, LS, J); // 设置线性求解器
    if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn); // 设置回调

    // 执行求解
    // int flag = KINSol(kin_mem, u, KIN_NONE, u, u); // 使用纯牛顿法求解
    int flag = KINSol(kin_mem, u, KIN_LINESEARCH, u, u); // 使用线搜索求解
    bool is_success = (flag >= 0);

    // 获取结果
    py::array_t<double> x_res(N);
    for (int i = 0; i < N; ++i) x_res.mutable_data()[i] = data.lb[i] + NV_Ith_S(u, i);
    
    double fnorm;
    KINGetFuncNorm(kin_mem, &fnorm); // 返回最终残差的范数

    N_VDestroy(u); N_VDestroy(constr); SUNLinSolFree(LS); SUNMatDestroy(J);
    KINFree(&kin_mem); SUNContext_Free(&sunctx);

    // 指针返回解向量、残差范数、求解是否成功、KINSol返回值
    return py::dict("x"_a=x_res, "fun"_a=fnorm, "success"_a=is_success, "status"_a=flag);
}

PYBIND11_MODULE(kinsol_cpp, m) { // 定义一个 Python 模块，名为 kinsol_cpp
    // m.def() 是 py::module_ 的方法，用于绑定 C++ 函数到 Python。
    // 第一个参数 "solve" 是 Python 中调用的函数名。
    // 第二个参数 &solve_cpp 是 C++ 函数的指针（即要暴露的函数）。
    m.def("solve", &solve_cpp);
}