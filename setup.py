from setuptools import setup, find_packages

setup(
    name="aitools",
    version=0.3,
    packages=find_packages("library"),
    package_dir={"": "library"},
)
