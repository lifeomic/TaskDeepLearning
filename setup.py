from setuptools import setup
from setuptools import find_packages

setup(name='taskdl',
      version='0.1.4',
      description='Deep learning on Lifeomic',
      keywords = ['tensorflow', 'machine learning', 'lifeomic', 'deep learning', 'pytorch'],
      url='https://github.com/lifeomic/TaskDeepLearning',
      download_url='https://github.com/lifeomic/TaskDeepLearning/archive/0.1.4.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      install_requires=['requests'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False )