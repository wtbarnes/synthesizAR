import datetime

from packaging.version import Version

# -- Project information -----------------------------------------------------

project = 'synthesizAR'
author = 'Will Barnes'
copyright = f'{datetime.datetime.now().year}, {author}'

# The full version, including alpha/beta/rc tags
from synthesizAR import __version__

_version = Version(__version__)
version = release = str(_version)
# Avoid "post" appearing in version string in rendered docs
if _version.is_postrelease:
    version = release = _version.base_version
# Avoid long githashes in rendered Sphinx docs
elif _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
is_development = _version.is_devrelease
is_release = not(_version.is_prerelease or _version.is_devrelease)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.bibtex',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'
default_role = 'obj'
napoleon_use_rtype = False
napoleon_google_docstring = False

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'numpy': ('https://numpy.org/doc/stable/',
              (None, 'http://data.astropy.org/intersphinx/numpy.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/',
              (None, 'http://data.astropy.org/intersphinx/scipy.inv')),
    'matplotlib': ('https://matplotlib.org/',
                   (None, 'http://data.astropy.org/intersphinx/matplotlib.inv')),
    'astropy': ('https://docs.astropy.org/en/stable', None),
    'sunpy': ('https://docs.sunpy.org/en/stable/', None),
    'aiapy': ('https://aiapy.readthedocs.io/en/stable/', None),
    'xrtpy': ('https://xrtpy.readthedocs.io/en/stable/', None),
    'fiasco': ('https://fiasco.readthedocs.io/en/stable/', None),
    'pydrad': ('https://pydrad.readthedocs.io/en/latest/', None),
    'ebtelplusplus': ('https://ebtelplusplus.readthedocs.io/en/stable', None),
    'dask': ('https://docs.dask.org/en/latest', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_nav_level": 2,
    "logo": {
        "text": f"synthesizAR {version}",
    },
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wtbarnes/synthesizAR",
            "icon": "fa-brands fa-github",
        },
    ],
}
html_context = {
    "github_user": "wtbarnes",
    "github_repo": "synthesizAR",
    "github_version": "main",
    "doc_path": "docs",
}

bibtex_bibfiles = ['references.bib']

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    '-Nfontsize=10',
    '-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Efontsize=10',
    '-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif',
    '-Gfontsize=10',
    '-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif'
]


# -- Sphinx gallery -----------------------------------------------------------
extensions += ['sphinx_gallery.gen_gallery']
sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'generated/gallery',  # path to where to save gallery generated output
    'filename_pattern': '^((?!skip_).)*$',
    'default_thumb_file': '_static/synthesizar_logo.png',
    'matplotlib_animations': True,
}
