from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='CG',
    ext_modules=[
        CppExtension(
            'CG',
            [
                'main.cpp'
                ''
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
