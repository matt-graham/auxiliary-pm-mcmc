# -*- coding: utf-8 -*-
"""
Setup script for auxiliary pseudo-marginal MCMC python implementations and
Gaussian process classification parameter inference experiments.
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

from distutils.core import setup
from distutils.extension import Extension
import argparse
import sys
import os

parser = argparse.ArgumentParser(
    description='Auxiliary pseudo-marginal MCMC python implementations')
parser.add_argument('-debug', action='store_true', default=False,
                    help='Export GDB debug information when compiling')
parser.add_argument('-use-cython', action='store_true', default=False,
                    help='Use Cython to compile from .pyx files')
parser.add_argument('-use-gcc-opts', action='store_true', default=False,
                    help='Use GCC compiler optimisations for quicker but '
                         'less safe mathematical operations')
parser.add_argument('-use-cython-opts', action='store_true', default=False,
                    help='Add extra Cython compile directives for quicker '
                         'but less safe array access. Requires -use-cython '
                         'to be set to have an effect.')

# hack to get both argparser help and distutils help displaying
if '-h' in sys.argv:
    sys.argv.remove('-h')
    help_flag = '-h'
elif '--help' in sys.argv:
    sys.argv.remove('--help')
    help_flag = '--help'
else:
    help_flag = None

args, unknown = parser.parse_known_args()

# remove custom arguments from sys.argv to avoid conflicts with distutils
for action in parser._actions:
    for opt_str in action.option_strings:
        try:
            sys.argv.remove(opt_str)
        except ValueError:
            pass
# if a help flag was found print parser help string then readd so distutils
# help also displayed
if help_flag:
    parser.print_help()
    sys.argv.append(help_flag)

ext = '.pyx' if args.use_cython else '.c'

extra_compile_args = ['-O3', '-ffast-math'] if args.use_gcc_opts else []


if args.use_cython and args.use_cython_opts:
    compiler_directives = {
        'boundscheck': False,  # don't check for invalid indexing
        'wraparound': False,  # assume no negative indexing
        'cdivision': True,  # don't check for zero division
        'initializedcheck': False,  # don't check memory view init
        'embedsignature': True  # include call signature in docstrings
    }
else:
    compiler_directives = {}

ext_modules = [
    Extension('gpdemo.kernels',
              [os.path.join('gpdemo', 'kernels') + ext],
              extra_compile_args=extra_compile_args)
]

if args.use_cython:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules,
                            compiler_directives=compiler_directives,
                            gdb_debug=args.debug)

setup(
    name='Auxiliary pseudo-marginal MCMC python implementations',
    author='Matt Graham',
    packages=['auxpm', 'gpdemo'],
    ext_modules=ext_modules,
)
