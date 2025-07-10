from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Detect PyTorch C++ ABI
cxx11_abi = int(torch.compiled_with_cxx11_abi())
abi_flag = f'-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}'
print(f"[setup.py] Using ABI flag: {abi_flag}")

setup(
    name='project_features_cuda',
    version='1.0',
    ext_modules=[
        CUDAExtension(
            'project_features_cuda',
            ['project_image_cuda.cpp',
             'project_image_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', abi_flag],
                'nvcc': ['-O3', '-std=c++17', abi_flag]
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)