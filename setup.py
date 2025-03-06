#!/usr/bin/env python
import setuptools
setuptools.setup(
    packages=setuptools.find_packages() + ['pyepri', 'pyepri.datasets'],
    package_dir={
        'pyepri': 'src/pyepri',
        'pyepri.datasets': 'datasets',
    },
    package_data={'datasets': ['*.npy', '*.txt', '*.DSC', '*.DTA']},
)

