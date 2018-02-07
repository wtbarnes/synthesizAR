===============
Getting Started
===============

Dependencies
------------
synthesizAR is compatible with Python 3.6+. The easiest and most convenient way to install Python is through the `Anaconda <https://www.continuum.io/downloads>`_ distribution. Additionally, synthesizAR requires several other packages from the scientific Python ecosystem,

- astropy
- dask
- h5py
- fiasco
- matplotlib
- numpy
- numba
- plasmapy
- scipy
- sunpy
- yt

These can be installed through either `conda` (recommended) or pip.

Install
-------
To download synthesizAR and install all the needed dependencies (with `conda`), and then install all of the needed dependencies,

.. code-block:: bash

   $ git clone https://github.com/wtbarnes/synthesizAR.git
   $ cd synthesizAR

The easiest way to grab all of the dependencies is to create a new conda-environment and install them into there. This can be done easily with the included environment file,

.. code-block:: bash

   $ conda env create -f conda_environment.yml
   $ source activate synthesizar

This will create a new `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ with the needed dependencies and activate the environment. Finally, to install synthesizAR,

.. code-block:: bash

   $ python setup.py install

Updating
--------
As synthesizAR is not yet available on any package manager, the best way to keep up with releases is to pull down updates from GitHub. To grab the newest version from GitHub and install it, inside the package repo,

.. code-block:: bash

   $ git pull
   $ python setup.py install

If you'd like to maintain a fork of synthesizAR (e.g. if you need to make changes to the codebase or contribute to the package), see the :doc:`Dev page </develop>`.
