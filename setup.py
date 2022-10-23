from setuptools import setup, find_packages

# TODO reading version from version file does not work
# VERSION = {}  # type: ignore
# with open("version.py", "r") as version_file:
#    exec(version_file.read(), VERSION)

setup(name='mesocircuit',
      version='0.1.0',
      description='Mesocircuit Model',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/INM-6/mesocircuit-model',
      author='see docs/source/authors.rst',
      license='GNU GPLv3',
      packages=find_packages(include=['mesocircuit', 'mesocircuit.*']),
      install_requires=[],
      python_requires='>=3')
