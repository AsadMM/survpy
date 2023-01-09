# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:15:44 2023

@author: asadm
"""

import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="survpy",
    version="0.0.1",
    author="Asad Mujawar",
    author_email="mujawar.asad.98@gmail.com",
    description="A Python library with dumb code for analyzing surveys.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-Clause 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)