from setuptools import find_packages, setup

setup(author="Ian L. Morgan",
      version="0.1.1",
      name="stepy",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=['lmfit',
                        'matplotlib',
                        'numpy>1.20',
                        'scipy']
      )
