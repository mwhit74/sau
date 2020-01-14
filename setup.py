from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sau',
      version='0.1',
      description='Structural analysis utilities',
      long_description=readme(),
      license='MIT',
      author='mlw',
      url='https://github.com/mwhit74/sau',
      packages=[],
      install_requires=[])
