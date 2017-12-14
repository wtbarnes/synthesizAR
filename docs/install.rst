=============
Installation
=============

synthesizAR is built on Python 3 and has not been tested on Python 2. The easiest and most convenient way to install Python is through the `Anaconda <https://www.continuum.io/downloads>`_ package manager. Additionally, synthesizAR requires several other packages from the scientific Python ecosystem,

- AstroPy
- ChiantiPy
- h5py
- matplotlib
- NumPy
- numba (optional)
- scipy
- solarbextrapolation
- SunPy
- wcsaxes
- YT

Download
=========
synthesizAR is freely available and developed in the open on GitHub. To download synthesizAR, install all the needed dependencies (with `conda`), and then install all of the needed dependencies,

.. code-block:: bash

   $ git clone git@github.com:wtbarnes/synthesizAR.git
   $ cd synthesizAR
   $ conda env create -f conda_environment.yml
   $ source activate synthesizar
   $ python setup.py install

This will clone the git repository, create a new `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ with the needed dependencies, activate the environment, and then install synthesizAR into this new environment. If you did not use Anaconda to install Python or prefer to manage your environment in a different way, you can of course install these dependencies by hand via `conda` or `pip`.

Updating
=========
As synthesizAR is not yet available on any package manager, the best way to keep up with releases is to pull down updates from GitHub. To grab the newest version from GitHub and install it,

.. code-block:: bash

   $ cd synthesizAR
   $ git pull
   $ python setup.py install

If you'd like to maintain a fork of synthesizAR (e.g. if you need to make changes to the codebase or contribute to the package), see the :doc:`Dev page </develop>`.
