from setuptools import find_packages, setup

REQUIREMENTS_PATH = "requirements.txt"


with open(REQUIREMENTS_PATH) as our_file:
    required_libraries = our_file.read().splitlines()

setup(
    name="Homework02",
    version="0.1.0",
    description="Machine learning in production. Second homework.",
    long_description="README.md",
    packages=find_packages(),
    author="Khusainov Eldar",
    install_requires=required_libraries,
    license="MIT",
    license_files="LICENSE",
)
