from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT="-e ."
def get_requirements(file_path:str)->List[str ]:
    # return the requirements list
    requirements=[]
    with open(file_path) as object:
        requirements=object.readlines()
        requirements = [packages.replace("\n","") for packages in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="mlProject",
    version="0.0.1",
    author="Navneet",
    author_email="officialnavneetdas@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)