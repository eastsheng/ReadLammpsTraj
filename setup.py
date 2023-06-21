"""run this
python setup.py sdist
pip install .
"""

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt","r") as f:
    required = f.read().splitlines()


setup(
name         = 'ReadLammpsTraj',
version      = '1.1.5',
py_modules   = ['ReadLammpsTraj'],
author       = 'CHENDONGSHENG',
author_email = 'eastsheng@hotmail.com',
packages=find_packages('src'),
package_dir={'': 'src'},
install_requires=required,
url          = 'https://github.com/eastsheng/ReadLammpsTraj',
description  = 'Read lammps dump trajectory.',
long_description=long_description,
long_description_content_type='text/markdown'
)

