from setuptools import setup, find_packages

setup(
    name='medseg_evaluator',
    version='0.1.0',
    author='Darshan Dathiya',
    author_email='darshandathiya2@gmail.com',
    description='Evaluation toolkit for medical image segmentation models',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-image',
        'nibabel',
        'pydicom'
    ],
)
