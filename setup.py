from setuptools import setup, find_packages

def get_requirements(file_path):
    """This function will return the list of requirements"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "")  for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name='MLproject',
    version='0.0.1',
    author='Majdi Habibi',
    author_email='majdi.hbibi.7@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(file_path= 'requirements.txt')
)

# ['pandas', 'numpy', 'seaborn', 'matplotlib', 'scikit-learn', 'catboost', 'xgboost', 'Flask']