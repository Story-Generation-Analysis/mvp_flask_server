from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# Define the extension
ext_modules = [
    Extension(
        name="cythonloop", 
        sources=["svm_optimized.pyx"], 
        libraries=["m"],  # Link against the math library (libm)
        extra_compile_args=["-O3","-march=native", "-ffast-math","-fopenmp"],  # Optional: optimization flags
        include_dirs=[],  # Optionally specify include directories for headers
    )
]


setup(
    ext_modules=cythonize(
        "svm_optimized.pyx",annotate=True,
            compiler_directives={
            "language_level": "3",
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,  # Enable C division (faster division without checks)
            'embedsignature': True,  # Enable function signature optimization
        }, 
    ),
    include_dirs=[numpy.get_include()]  # This includes the NumPy headers

)