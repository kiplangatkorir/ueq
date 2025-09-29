from setuptools import setup, find_packages

setup(
    name="ueq",
    version="1.0.1",
    description="Uncertainty Everywhere (UEQ) - Phoenix Edition: A unified Python library for Uncertainty Quantification with production-ready features",
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="uncertainty-quantification,machine-learning,deep-learning,bootstrap,conformal-prediction,mc-dropout,production,monitoring,auto-detection,cross-framework,phoenix",
    project_urls={
        "Source": "https://github.com/kiplangatkorir/ueq",
        "Bug Reports": "https://github.com/kiplangatkorir/ueq/issues",
    },
    license="Apache-2.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
