# -*- coding: utf-8 -*-

# ...header comments removed for anonymization...

import io
import os

from setuptools import setup

# Package meta-data.
NAME = 'smplx'
DESCRIPTION = 'PyTorch module for loading the SMPLX body model'
URL = ''
EMAIL = ''
AUTHOR = ''
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.28'

here = os.path.abspath(os.path.dirname(__file__))

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

pyrender_reqs = ['pyrender>=0.1.23', 'trimesh>=2.37.6', 'shapely']
matplotlib_reqs = ['matplotlib']
open3d_reqs = ['open3d-python']

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      install_requires=[
          'numpy>=1.16.2',
          'torch>=1.0.1.post2',
      ],
      extras_require={
          'pyrender': pyrender_reqs,
          'open3d': open3d_reqs,
          'matplotlib': matplotlib_reqs,
          'all': pyrender_reqs + matplotlib_reqs + open3d_reqs
      },
      packages=['smplx'])
