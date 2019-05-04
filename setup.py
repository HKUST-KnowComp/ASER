from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='aser',
    version='0.1.0',
    description='A Large-scale Eventuality Knowledge Graph',
    long_description=readme,
    license=license,
    python_requires='>=3',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)