# setup.py
from setuptools import setup, find_packages

setup(
    name="demuxnet",
    version="1.1.1",
    author="You Wu",
    author_email="wuyou1990@sjtu.edu.cn",
    description="Machine learning augmented sample demultiplexing of pooled single-cell RNA-seq data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/paddi1990/DemuxNet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List any required dependencies here
        "requests",
        "numpy",
        "rpy2"
    ],
    entry_points={
        "console_scripts": [
            "demuxnet=demuxnet.__main__:main",
        ],
    },
)
