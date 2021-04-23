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
    version='2.0.0',
    description='The higher-order selectional preference over collected linguistic graphs reflects various kinds of commonsense knowledge.',
    long_description=readme,
    license=license,
    python_requires='>=3',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
    entry_points={
        'console_scripts': ['aser-server=aser.server.cli:main',
                            'aser-pipe=aser.pipe.cli:main'],
    },
)
