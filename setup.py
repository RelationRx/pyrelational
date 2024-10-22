from typing import Dict

from setuptools import find_packages, setup

"""
pip install -e .
"""

setup_requires = ["pytest-runner"]
tests_require = ["pytest", "pytest-cov", "mock"]

with open("requirements/base_requirements.txt", "r") as req:
    install_requires = [line.strip() for line in req if line.strip()]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version: Dict[str, str] = {}
with open("pyrelational/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="pyrelational",
    description="Python tool box for quickly implementing active learning strategies",
    author="Relation Therapeutics",
    author_email="software@relationrx.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RelationRx/pyrelational",
    packages=find_packages(),
    version=version["__version__"],
    setup_requires=setup_requires,
    tests_require=tests_require,
    install_requires=install_requires,
    python_requires=">=3.9",
)
