cmake_minimum_required(VERSION 3.12)
project(pykinsol) # 建议与项目名保持一致

# 保持 C++14 标准以兼容 SUNDIALS v6.x
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 pybind11
find_package(pybind11 REQUIRED)

# 2. 寻找 SUNDIALS
# 如果你安装在非标准目录，可以取消下面这一行的注释并指向你的安装路径
# set(SUNDIALS_DIR "/usr/local/lib/cmake/sundials")
find_package(SUNDIALS REQUIRED COMPONENTS kinsol nvecserial sunmatrixdense sunlinsoldense)

# 3. 定义模块
# 【关键修改】：源文件路径更新为新结构下的路径
pybind11_add_module(kinsol_cpp pykinsol/src/kinsol_wrapper.cpp)

# 4. 链接库
target_link_libraries(kinsol_cpp PRIVATE 
    SUNDIALS::kinsol 
    SUNDIALS::nvecserial 
    SUNDIALS::sunmatrixdense 
    SUNDIALS::sunlinsoldense
)

# 5. 显式包含目录（确保头文件能被找到）
target_include_directories(kinsol_cpp PRIVATE 
    ${SUNDIALS_INCLUDE_DIRS}
    pykinsol/include
)