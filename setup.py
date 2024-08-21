#!/usr/bin/env python3
import setuptools

with open("README.md", encoding="utf8") as f:
    readme = f.read()

DISTNAME = "sem_objnav"
DESCRIPTION = "Object Search in Habitat"
LONG_DESCRIPTION = readme
AUTHOR = "Sai Prasanna"
LICENSE = "GPLv3"


if __name__ == "__main__":
    # package data are the files and configurations included in the package

    setuptools.setup(
        name=DISTNAME,
        install_requires=[],
        extras_require={
            "habitat": ["habitat-sim", "habitat-lab"],
        },
        packages=["sem_objnav"],
        version="0.1.0",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        license=LICENSE,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: MacOS",
            "Operating System :: Unix",
        ],
    )
