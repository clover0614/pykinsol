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

    // -----------------------------------------------------------------------------
    // [修改版调试逻辑] 放在 KinsolSysFn 函数中，填充完 f_data 之后，return 0 之前
    // -----------------------------------------------------------------------------

    static int print_count = 0;
    // 只看初值（第一次）的s1和s2
    if (print_count < 1) { 
        std::cout << "\n================ [DEBUG: Residual & Bound Check (Call " << print_count << ")] ================" << std::endl;
        std::cout << "Thresholds: Abs(Res) > 1e-3  OR  s < 1e-6" << std::endl;
        
        // 设置输出格式为科学计数法，保留 16 位有效数字
        // 这样能看清 5.578e-14 和 0.0 的区别，以及 0.1000000000000001 和 0.1 的区别
        std::cout << std::scientific << std::setprecision(16);

        bool found_issue = false;
        double max_phys_res = 0.0;
        double max_constr_res = 0.0;
        int max_phys_idx = -1;
        
        for (int i = 0; i < N; ++i) {
            double s1 = s_data[i];
            double s2 = s_data[i + N];
            double phys_res = f_data[i];       
            double constr_res = f_data[i + N]; 
            
            // 记录最大误差
            if (std::abs(phys_res) > max_phys_res) {
                max_phys_res = std::abs(phys_res);
                max_phys_idx = i;
            }
            if (std::abs(constr_res) > max_constr_res) max_constr_res = std::abs(constr_res);

            // 筛选条件
            bool is_res_bad = (std::abs(phys_res) > 1e-3) || (std::abs(constr_res) > 1e-3);
            bool is_bound_active = (s1 < 1e-6) || (s2 < 1e-6);

            if (is_res_bad || is_bound_active) {
                found_issue = true;
                std::cout << "[Idx " << i << "] ";
                
                if (is_res_bad) std::cout << "❌RES_FAIL ";
                if (is_bound_active) std::cout << "⚠️ON_BOUND ";
                
                std::cout << "| PhysRes=" << phys_res 
                        << " | ConstrRes=" << constr_res 
                        << " | s1=" << s1 
                        << " | s2=" << s2 
                        << " | lb=" << data->lb[i]
                        << " | ub=" << data->ub[i]
                        // 这里特意把 s1 单独加进去看，防止 lb 把 s1 的精度“吃掉”导致看不清
                        << " | x(lb+s1)=" << (data->lb[i] + s1)
                        << std::endl;
            }
        }
        
        if (!found_issue) {
            std::cout << "✅ No obvious outliers found." << std::endl;
        } else {
            std::cout << "----------------------------------------------------------------" << std::endl;
            std::cout << "📈 Max Phys Res: " << max_phys_res << " (at Idx " << max_phys_idx << ")" << std::endl;
            std::cout << "📈 Max Constr Res: " << max_constr_res << std::endl;
        }
        
        // 恢复默认输出格式（可选，以免影响后续日志）
        std::cout.unsetf(std::ios_base::floatfield); 
        std::cout << std::setprecision(6);
        std::cout << "======================================================================\n" << std::endl;
        
        print_count++;
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

// 主求解器实现
py::dict solve_cpp_impl(py::function func, py::array_t<double> x0, py::object jac, 
                        py::array_t<double> lb, py::array_t<double> ub,
                        int strategy, int linsol_type, 
                        int verbose) { // <---【新增】verbose 参数
    
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

    // 初始化松弛变量 u (保持您原来的松弛变量逻辑)
    N_Vector u = N_VNew_Serial(2 * N, sunctx);
    auto x0_p = x0.unchecked<1>();
    for (int i = 0; i < N; ++i) {
        NV_Ith_S(u, i) = x0_p(i) - data.lb[i];      // s1
        NV_Ith_S(u, i + N) = data.ub[i] - x0_p(i);  // s2
    }

    // 创建 KINSOL 内存
    void* kin_mem = KINCreate(sunctx);
    KINInit(kin_mem, KinsolSysFn, u);
    KINSetUserData(kin_mem, &data);

    // 设置约束：松弛变量必须大于等于0
    N_Vector constr = N_VNew_Serial(2 * N, sunctx);
    N_VConst(2.0, constr); 
    KINSetConstraints(kin_mem, constr);

    // --- 【新增】设置日志打印级别 ---
    // level 0: 不输出
    // level 1: 输出每次非线性迭代的统计信息 (残差范数, 步长)
    // level 3: 输出非常详细的调试信息 (包括线性求解细节)
    KINSetPrintLevel(kin_mem, verbose);

    // 配置线性求解器 (保持不变)
    SUNMatrix J = nullptr;
    SUNLinearSolver LS = nullptr;

    if (linsol_type == LINSOL_DENSE) {
        J = SUNDenseMatrix(2 * N, 2 * N, sunctx);
        LS = SUNLinSol_Dense(u, J, sunctx);
        KINSetLinearSolver(kin_mem, LS, J);
        if (data.has_jac) KINSetJacFn(kin_mem, KinsolJacFn);
    } 
    else if (linsol_type == LINSOL_GMRES) {
        LS = SUNLinSol_SPGMR(u, PREC_NONE, 0, sunctx);
        KINSetLinearSolver(kin_mem, LS, nullptr);
    }

    N_Vector scaling = N_VNew_Serial(2 * N, sunctx);
    N_VConst(1.0, scaling); 
    
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
                           std::string method, std::string linear_solver, 
                           int verbose) { // <--- 【新增】暴露给 Python
    
    int strategy_int = get_strategy_enum(method);
    int linsol_int = get_linsol_enum(linear_solver);

    return solve_cpp_impl(func, x0, jac, lb, ub, strategy_int, linsol_int, verbose);
}

// 模块定义
PYBIND11_MODULE(pykinsol, m) {
    m.doc() = "Kinsol solver with logging control";

    m.def("pykinsol", &solve_cpp_wrapper, 
          "func"_a, 
          "x0"_a, 
          "fprime"_a = py::none(), 
          "lb"_a = py::none(), 
          "ub"_a = py::none(), 
          "method"_a = "linesearch", 
          "linear_solver"_a = "dense",
          "verbose"_a = 1 // <--- 【新增】默认开启 1 级日志
    );
    
    m.attr("KIN_NONE") = py::int_(KIN_NONE);
    m.attr("KIN_LINESEARCH") = py::int_(KIN_LINESEARCH);
}