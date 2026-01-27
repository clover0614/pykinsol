import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 自定义构建类，处理不同操作系统的差异
class BuildExt(build_ext):
    def build_extensions(self):
        # 针对 Windows 的特殊编译器参数
        if self.compiler.compiler_type == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args += ["/O2", "/EHsc"]
        super().build_extensions()

# 获取 SUNDIALS 路径（由 GitHub Actions 设置）
sundials_root = os.environ.get("SUNDIALS_ROOT", "/usr/local")

ext_modules = [
    Pybind11Extension(
        "pykinsol.kinsol_cpp",
        ["pykinsol/src/kinsol_wrapper.cpp"],
        include_dirs=[
            os.path.join(sundials_root, "include"),
        ],
        library_dirs=[
            os.path.join(sundials_root, "lib"),
            os.path.join(sundials_root, "lib64"),
        ],
        # 【关键修改】在 libraries 中增加 "sundials_sunlinsolspgmr"
        # 注意：sundials_sunlinsoldense 也建议显式写出来，以防万一
        libraries=[
            "sundials_kinsol", 
            "sundials_nvecserial", 
            "sundials_sunmatrixdense",  # 对应 SUNDenseMatrix
            "sundials_sunlinsoldense",  # 对应 SUNLinSol_Dense
            "sundials_sunlinsolspgmr"   # [新增] 对应 SUNLinSol_SPGMR (GMRES)
        ],
        cxx_std=14,
        extra_compile_args=["-Wall", "-O2"],
    ),
]

setup(
    name="pykinsol",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    packages=["pykinsol"],
)
