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
        "pykinsol", # 【修改1】这里直接叫 "pykinsol"，生成的 pyd 文件就会在顶层
        ["pykinsol/src/kinsol_wrapper.cpp"], # 源码路径保持不变
        include_dirs=[
            os.path.join(sundials_root, "include"),
        ],
        library_dirs=[
            os.path.join(sundials_root, "lib"),
            os.path.join(sundials_root, "lib64"),
        ],
        libraries=[
            "sundials_kinsol", 
            "sundials_nvecserial", 
            "sundials_sunmatrixdense", 
            "sundials_sunlinsoldense", 
            "sundials_sunlinsolspgmr"
        ],
        cxx_std=14,
        extra_compile_args=["-Wall", "-O2"],
    ),
]

setup(
    name="pykinsol",
    version="0.4.0", # 建议升级版本号
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    packages=[], # 设为空列表，表示不包含任何 Python 源码包
)