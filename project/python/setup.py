"""
Build script for the CUDA Batched Causal Ablation
Engine Python extension.

Usage:
  pip install -e .
  python -c "import ablation_engine; ..."
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension, CUDAExtension
)

# Source files for the extension
src_dir = os.path.join(
    os.path.dirname(__file__), '..', 'src')

sources = [
    'ablation_engine.cpp',
    os.path.join(src_dir, 'transformer.cu'),
    os.path.join(src_dir, 'ablation.cu'),
    os.path.join(src_dir, 'baseline.cu'),
    os.path.join(src_dir, 'batched_engine.cu'),
    os.path.join(src_dir, 'benchmark.cu'),
    os.path.join(src_dir, 'validation.cu'),
    os.path.join(src_dir, 'weight_io.cu'),
]

inc_dir = os.path.join(
    os.path.dirname(__file__), '..', 'include')

setup(
    name='ablation_engine',
    version='0.2.0',
    description='CUDA Batched Causal Ablation Engine',
    ext_modules=[
        CUDAExtension(
            name='ablation_engine',
            sources=sources,
            include_dirs=[inc_dir, src_dir],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '--expt-extended-lambda',
                    '-std=c++14',
                ],
            },
            libraries=['cublas'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
