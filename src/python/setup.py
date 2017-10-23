#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [Extension(
    'sssfile',
    sources = ['bindings/main.cpp'],
    libraries = ['sssfile'],
    extra_compile_args=["-O3"],
)]

setup(
    name = 'sssfile',
    version = '0',
    include_dirs = [np.get_include(), '../sssfile/include/'],
    ext_modules = ext_modules
    )
