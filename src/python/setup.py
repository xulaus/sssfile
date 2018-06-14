#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import setuptools
import numpy as np

from distutils.core import setup, Extension

EXT_MODULES = [Extension(
    'sssfile',
    sources=['bindings/main.cpp'],
    libraries=['sssfile'],
    extra_compile_args=["-O3"],
)]

setup(
    name='sssfile',
    version='0',
    include_dirs=[np.get_include(), '../sssfile/include/'],
    ext_modules=EXT_MODULES
)
