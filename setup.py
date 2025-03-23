from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transformer-implementation",
    version="0.1.0",
    author="Keshav Rajbux",
    author_email="your.email@example.com",
    description="PyTorch implementation of the Transformer architecture from 'Attention Is All You Need'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keshavrajbux/AttentionIsAllYouNeed",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "torchtext>=0.10.0",
        "spacy>=3.0.0",
    ],
) 