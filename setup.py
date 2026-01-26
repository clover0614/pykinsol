import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
        # 静态链接 SUNDIALS 的核心组件
        libraries=["sundials_kinsol", "sundials_nvecserial"],
        cxx_std=14,
    ),
]

setup(
    name="pykinsol",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    packages=["pykinsol"],
)