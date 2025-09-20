from setuptools import setup, find_packages

setup(
    name="ueq",
    version="0.1.0",
    description="Uncertainty Everywhere - A unified Python library for Uncertainty Quantification",
    author="Kiplangat Korir",
    author_email="korirkiplangat22@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "pylint>=2.8.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="uncertainty-quantification machine-learning deep-learning bootstrap conformal-prediction mc-dropout",
    project_urls={
        "Source": "https://github.com/kiplangatkorir/ueq",
        "Bug Reports": "https://github.com/kiplangatkorir/ueq/issues",
    },
)