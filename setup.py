from setuptools import setup, find_packages

setup(name='deepmol',
      version='1.0',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      author='BiSBII CEB University of Minho',
      author_email='jfscorreia95@gmail.com',
      description='DeepMol: ...',
      license='BSD-2-Clause License',
      keywords='machine-learning deep-learning chemoinformatics',
      url=''
      )