from setuptools import setup, find_packages

setup(name='mesocircuit',
      version='1.0',
      description='Mesocircuit Model',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/INM-6/mesocircuit-model',
      author='see docs/source/authors.rst',
      license='GNU GPLv3',
      packages=find_packages(include=['mesocircuit', 'mesocircuit.*']),
      install_requires=[],
      python_requires='>=3')
