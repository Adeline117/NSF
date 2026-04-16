"""setup.py for onchain_audit.

Install in-tree with:
    pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="onchain_audit",
    version="0.1.0",
    description="Three-step leakage audit for on-chain agent-vs-human classifiers.",
    author="Adeline Wen and co-authors",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "pyarrow>=12.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "jupyter>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "onchain-audit-example=onchain_audit.example:main",
        ],
    },
)
