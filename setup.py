from setuptools import find_packages, setup

setup(
    name='model_risk_management',
    version='1.0',
    author='Antoine Amend',
    author_email='antoine.amend@databricks.com',
    description='Generate documentation for model risk management',
    include_package_data=True,
    install_requires=[
        'requests==2.31.0',
        'PyYAML==6.0',
        'mdtex2html==1.2.0',
        'graphviz==0.20.1',
        'pdfkit==1.0.0',
        'mlflow==2.11.1'
    ],
    long_description_content_type='text/markdown',
    url='https://github.com/databricks-industry-solutions/fsi-mrm-generation',
    packages=find_packages(where='.'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: Other/Proprietary License',
    ],
)
