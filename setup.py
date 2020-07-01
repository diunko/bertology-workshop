
from setuptools import find_namespace_packages, setup

setup_args = dict(
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src")
)

setup(**setup_args)