from setuptools import setup, find_packages

setup(
    name="aitools",
    version=0.1,
    packages=find_packages("sample"),
    package_dir={"": "sample"},
)
