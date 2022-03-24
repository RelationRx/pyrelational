from setuptools import find_packages, setup

"""
pip install -e .
"""

setup_requires = ["pytest-runner"]
tests_require = ["pytest", "pytest-cov", "mock"]

install_requires = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "pytorch-lightning>=1.5",
    "torch>=1.9.0",
    "scikit-learn>=1.0.2",
    "tabulate>=0.7.0",
]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyrelational",
    description="Python tool box for quickly implementing active learning strategies",
    author="Relation Therapeutics",
    author_email="software@relationrx.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RelationRx/pyrelational",
    packages=find_packages(),
    version="0.1.4",
    setup_requires=setup_requires,
    tests_require=tests_require,
    install_requires=install_requires,
    python_requires=">=3.8",
)
