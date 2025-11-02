from setuptools import setup, find_packages

setup(
    name="custom_samplers",
    version="1.0.0",
    description="Adaptive Kernel Mix Sampling for NeRF",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "nerfstudio>=0.3.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
