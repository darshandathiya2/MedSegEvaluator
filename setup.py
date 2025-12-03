from setuptools import setup, find_packages

setup(
    name='medsegevaluator',  # Package name
    version='0.1.0',
    author='Darshan Dathiya',
    author_email='darshandathiya2@gmail.com',
    description='Evaluation toolkit for medical image segmentation models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/medsegevaluator',  # Replace with actual URL
    packages=find_packages(include=['medsegevaluator', 'medsegevaluator.*']),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Example: 'medseg-eval=medsegevaluator.cli:main',
        ],
    },
)
