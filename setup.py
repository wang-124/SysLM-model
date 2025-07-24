from setuptools import setup, find_packages

setup(
    name="syslm",
    version="0.1.0",
    description="A Systematic Longitudinal Study of Microbiome",
    author="Wang",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "scikit-learn==1.1.1",
        "matplotlib==3.7.2",
        "scipy==1.10.1",
        "torch==2.0.1+cu118",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
