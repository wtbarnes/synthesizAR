.. synthesizAR documentation master file

synthesizAR
============

synthesizAR is a Python package for building synthetic solar/stellar active regions comprised of coronal loops. The package allows one to build up a forward-modeled observation of an active region, starting with the magnetogram FITS file and ending with an observational data product. The following capabilities are included in synthesizAR:

- Extrapolation of 3D vector magnetic fields from magnetograms
- Coordination and configuration of hydrodynamic loop simulations from various models (e.g. EBTEL and HYDRAD)
- Synthesis of spectral lines for any number of ions/transitions using CHIANTI
- Mapping of emission to detector geometry

synthesizAR is designed to be modular and extensible. Extensions for new hydrodynamic codes, emission models, or instruments can be easily dropped in. The goal of synthesizAR is to provide an easy-to-use and efficient pipeline for forward-modeling hydrodynamic results from end to end. This will allow for easier comparison of model results to observations.

.. toctree::
   :maxdepth: 2

   getting_started
   code_ref/index
   examples
   develop
