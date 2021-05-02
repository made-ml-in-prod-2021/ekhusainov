import os
from setuptools import find_packages, setup

REQUIREMENTS_PATH = "requirements.txt"


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


with open(REQUIREMENTS_PATH) as our_file:
    required_libraries = our_file.read().splitlines()

setup(
    name="Homework01",
    version="0.0.1",
    description="Machine learning in production. First homework.",
    long_description=read("README.md"),
    packages=find_packages(),
    author="Khusainov Eldar",
    install_requires=required_libraries,
    license_files=read("LICENSE"),
)
