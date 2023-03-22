from setuptools import setup

# reading long description from file
with open('DESCRIPTION.txt') as file:
    long_description = file.read()

REQUIREMENTS = [
    "transformers>=4.11.3",
    "torch>=1.10",
    "numpy",
    "bert_score",
    "spacy",
]

# some more details
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]

# calling the setup function
setup(
    name='selfcheckgpt',
    version='0.1.1',
    description='SelfCheckGPT - Assessing text-based responses from LLMs',
    long_description=long_description,
    url='https://github.com/potsawee/selfcheckgpt',
    author='Potsawee Manakul',
    author_email='m.potsawee@gmail.com',
    license='MIT',
    packages=['selfcheckgpt'],
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords='selfcheckgpt',
    include_package_data=True,
)
