import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MedSegEvaluator'
author = 'MegSegEvaluator Comunity'
copyright = "2025, MedSegEvaluator community, http:github.com/darshandathiya2/MedSegEvaluator.git"
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'sphinx.ext.napoleon',  # for NumPy/Google style docstrings
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'class-doc-from': 'class'
}


# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["medsegevalutor."]

autosummary_generate = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Optional: If your code imports tensorflow or cv2 add this:
autodoc_mock_imports = [
    'tensorflow', 'keras', 'cv2', 'numpy', 'pandas', 'nibabel', 'pydicom'
]
