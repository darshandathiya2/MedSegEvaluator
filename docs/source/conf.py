import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'medsegevaluator'
author = 'Darshan Dathiya'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

autosummary_generate = True

# Optional: If your code imports tensorflow or cv2 add this:
autodoc_mock_imports = [
    'tensorflow', 'keras', 'cv2', 'numpy', 'pandas', 'nibabel', 'pydicom'
]
