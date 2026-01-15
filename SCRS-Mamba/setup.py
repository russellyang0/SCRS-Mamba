from setuptools import find_packages, setup

setup(
    name='scrsmamba',
    version='0.1.0',
    description='SCRS-Mamba for Remote Sensing Scene Classification',
    author='Zaichun Yang',
    license='Apache License 2.0',
    packages=find_packages(exclude=('configs', 'tools', 'data', 'datainfo', 'work_dirs', 'outputs')),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'mmengine>=0.8.3,<1.0.0',
        'mmcv>=2.0.0,<2.4.0',
        'mmpretrain>=1.2.0',
        'numpy',
        'matplotlib',
        'einops',
        'transformers>=4.39.0',
        'mamba-ssm',
        'causal-conv1d',
    ],
)
