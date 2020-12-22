#!/usr/bin/env python
'''meso_analysis setup.py file'''

from setuptools import setup, Extension
import numpy
from Cython.Distutils import build_ext
cmdclass = {'build_ext': build_ext}
ext_modules = [
    Extension('meso_analysis.helperfun',
              ['meso_analysis/helperfun.pyx'],
              include_dirs=[numpy.get_include()]),
]


with open('README.md') as file:
    long_description = file.read()


setup(
    name='meso_analysis',
    version='0.1',
    maintainer=['Espen Hagen', 'Johanna Senk'],
    maintainer_email=['2492641+espenhgn@users.noreply.github.com',
                      'j.senk@fz-juelich.de'],
    url='github.com/INM6',
    packages=['meso_analysis'],
    provides=['meso_analysis'],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    description='methods to analyze cortical mesocircuit network model',
    long_description=long_description,
    license='LICENSE',
)
