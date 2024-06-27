from setuptools import setup, find_packages



REQUIRED = [
    'torch',
    'flask',
    'shap',
    'numpy'
]

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='trainreaction',
    version='0.1.0',
    description='Web UI for graphing train loss and consulting chatbot.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Andy S. Yu',
    author_email='andy.sigeyu@gmail.com',
    url='',
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license='GNU GPLv3',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Natural Language :: English',
    ],
)
