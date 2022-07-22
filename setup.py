""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file


exec(open('version.py').read())
setup(
    name='jimm',
    version=__version__,
    description='(Unofficial) Jittor Image Models',
    long_description_content_type='text/markdown',
    url='https://github.com/Jittor-Image-Models/Jittor-Image-Models',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained models efficientnet mobilenetv3 mnasnet',
    packages=find_packages(exclude=['convert', 'tests', 'results']),
    include_package_data=True,
    install_requires=['torch >= 1.4', 'torchvision', 'scipy'],
    python_requires='>=3.6',
)
