import os
from setuptools import find_packages, setup


# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))
def find_package_requirements(filename):
    requirements = []
    with open(filename, 'r') as f:
        lines = [line for line in f.readlines()
                 if not (line.startswith('#') or line.startswith('-'))
                 ]
        return lines

def get_version():
    if os.path.isfile("VERSION"):
        with open('VERSION') as fh:
            return fh.read().strip()
    else:
        return "0.0"

setup(
    name='tamagotchi',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    license='Proprietary License',
    description='Tamagotchi application',
    long_description='',
    url='',
    author='Dewald Johan Abrie',
    author_email='dewaldabrie@gmail.com',
    platforms='Any',
    classifiers=[
        'Environment :: Desktop Environment',
        'Intended Audience :: Gamers',
        'License :: OSI Approved :: Proprietary License',  # example license
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=find_package_requirements('requirements-install.txt'),
)