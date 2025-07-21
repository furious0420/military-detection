#!/usr/bin/env python3
"""
Setup script for YOLO Multi-Class Detection project.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="military-detection",
    version="1.0.0",
    author="furious0420",
    author_email="furious0420@github.com",
    description="Military-grade object detection system with YOLOv8",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/furious0420/military-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    keywords="yolo, object-detection, computer-vision, pytorch, deep-learning, ai",
    project_urls={
        "Bug Reports": "https://github.com/furious0420/military-detection/issues",
        "Source": "https://github.com/furious0420/military-detection",
        "Documentation": "https://github.com/furious0420/military-detection/blob/main/README.md",
    },
)
