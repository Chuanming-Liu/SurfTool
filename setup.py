from setuptools import setup, find_packages


setup(
    name='SurfTool',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.6, <4',
    install_requires=[
        'matplotlib',
        'numpy',
        'obspy',
        'pyyaml',
        'setuptools',
        'tqdm',
    ],
    author='Chuanming Liu',
    author_email='chuanming.liu@colorado.edu',
    description='Surface wave  tool',
    license='MIT',
    url='https://github.com/Chuanming-Liu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    )
