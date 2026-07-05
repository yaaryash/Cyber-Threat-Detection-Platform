'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
to define the configuration of your project such as its 
metadata, dependencies, and more.
'''

from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    Reads requirements.txt and returns a list of dependencies,
    ignoring empty lines and the '-e .' editable install flag.
    """
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst


setup(
    name="CyberThreatDetectionPlatform",
    version="0.0.1",
    author="Yash",
    author_email="yashwalke3434@gmail.com",
    description="End-to-end MLOps pipeline for phishing website detection",
    packages=find_packages(),
    install_requires=get_requirements()
)