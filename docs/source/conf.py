import os
import sys

# ------------------------------------------------------------------------------
# Correct path so that Sphinx can find your package
# Your repo structure:
# MedSegEvaluator/
#     medsegevaluator/
#     docs/source/
# ------------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))    # Adds project root
sys.path.insert(0, os.path.abspath('../../medsegevaluator'))

# -- Project information -----------------------------------------------------

project = 'MedSegEvaluator'
author = 'MedSegEvaluator Community'
copyright = (
    "2025, MedSegEvaluator community, "
    "https://github.com/darshandathiya2/MedSegEvaluator"
)
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True

# ------------------------------------------------------------------------------
# Correct autodoc options (FINAL)
# ------------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",      # only useful special method
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# ------------------------------------------------------------------------------
# If your code imports heavy libraries (tf, cv2), mock them
# ------------------------------------------------------------------------------
autodoc_mock_imports = [
    'tensorflow', 'keras', 'cv2', 'numpy', 'pandas',
    'nibabel', 'pydicom', 'matplotlib'
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Improve RTD theme for long docs
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
}
