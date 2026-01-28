cmake_minimum_required(VERSION 3.10)
project(pykinsol)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 Pybind11
# 如果 cmake 找不到，运行时需指定: cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
find_package(pybind11 REQUIRED)

# 2. 设置 SUNDIALS 路径
# 优先读取环境变量 SUNDIALS_ROOT，默认值为 /usr/local
if(NOT DEFINED SUNDIALS_ROOT)
    if(DEFINED ENV{SUNDIALS_ROOT})
        set(SUNDIALS_ROOT $ENV{SUNDIALS_ROOT})
    else()
        set(SUNDIALS_ROOT "/usr/local")
    endif()
endif()

message(STATUS "Using SUNDIALS root: ${SUNDIALS_ROOT}")

# 3. 寻找 SUNDIALS 核心库与模块
find_library(SUNDIALS_KINSOL_LIB   NAMES sundials_kinsol     HINTS ${SUNDIALS_ROOT}/lib ${SUNDIALS_ROOT}/lib64)
find_library(SUNDIALS_NVEC_LIB     NAMES sundials_nvecserial HINTS ${SUNDIALS_ROOT}/lib ${SUNDIALS_ROOT}/lib64)

# 寻找 Dense 相关的库 (对应 linear_solver='dense')
find_library(SUNDIALS_MATDENSE_LIB NAMES sundials_sunmatrixdense HINTS ${SUNDIALS_ROOT}/lib ${SUNDIALS_ROOT}/lib64)
find_library(SUNDIALS_LINDENSE_LIB NAMES sundials_sunlinsoldense HINTS ${SUNDIALS_ROOT}/lib ${SUNDIALS_ROOT}/lib64)

# 【新增】寻找 GMRES 相关的库 (对应 linear_solver='gmres')
find_library(SUNDIALS_SPGMR_LIB    NAMES sundials_sunlinsolspgmr HINTS ${SUNDIALS_ROOT}/lib ${SUNDIALS_ROOT}/lib64)

# 检查是否都找到了
if (NOT SUNDIALS_SPGMR_LIB)
    message(WARNING "Could not find sundials_sunlinsolspgmr. GMRES solver might fail to link.")
endif()

# 4. 定义编译目标 (生成 .so/.pyd)
# 这里的 "kinsol_cpp" 必须和 PYBIND11_MODULE(kinsol_cpp, m) 里的名字一致
pybind11_add_module(kinsol_cpp pykinsol/src/kinsol_wrapper.cpp)

# 5. 添加头文件搜索路径
target_include_directories(kinsol_cpp PRIVATE
    ${SUNDIALS_ROOT}/include
)

# 6. 链接库文件
target_link_libraries(kinsol_cpp PRIVATE
    ${SUNDIALS_KINSOL_LIB}
    ${SUNDIALS_NVEC_LIB}
    ${SUNDIALS_MATDENSE_LIB} # 链接 Dense 矩阵库
    ${SUNDIALS_LINDENSE_LIB} # 链接 Dense 求解器库
    ${SUNDIALS_SPGMR_LIB}    # 【新增】链接 GMRES 库
)

# 打印调试信息
message(STATUS "Linked libraries:")
message(STATUS "  KINSOL: ${SUNDIALS_KINSOL_LIB}")
message(STATUS "  SPGMR:  ${SUNDIALS_SPGMR_LIB}")