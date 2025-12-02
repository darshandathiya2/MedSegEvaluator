import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../medsegevaluator/src'))

# -- Project information -----------------------------------------------------

project = 'MedSegEvaluator'
author = 'MegSegEvaluator Comunity'
copyright = "2025, MedSegEvaluator community, http:github.com/darshandathiya2/MedSegEvaluator.git"
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

autosummary_generate = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Optional: If your code imports tensorflow or cv2 add this:
autodoc_mock_imports = [
    'tensorflow', 'keras', 'cv2', 'numpy', 'pandas', 'nibabel', 'pydicom'
]
