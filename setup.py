import os
import sys
from collections import defaultdict
from glob import glob
from setuptools import setup, find_packages


def iter_files(path):
    """ Walk through all files located under a root path

    :param path: the directory path
    :type path: str
    :return: all file paths in this directory
    :rtype: List[str]
    """

    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

data_files = defaultdict(list)
for data_file in iter_files(os.path.join("aser", "extract", "discourse")):
    data_files[os.path.dirname(data_file)].append(data_file)
data_files = [(k, v) for k, v in data_files.items()]

setup(
    name='aser',
    version='2.0.0',
    description='The higher-order selectional preference over collected linguistic graphs reflects various kinds of commonsense knowledge.',
    long_description=readme,
    license=license,
    python_requires='>=3',
    packages=find_packages(exclude=('data')),
    data_files=data_files,
    install_requires=reqs.strip().split('\n'),
    entry_points={
        'console_scripts': ['aser-server=aser.server.cli:main',
                            'aser-pipe=aser.pipe.cli:main'],
    },
)
