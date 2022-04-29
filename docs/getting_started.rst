===============
Getting Started
===============

Install
-------
To download synthesizAR, first clone the GitHub repository,

.. code-block:: bash

   $ git clone https://github.com/wtbarnes/synthesizAR.git
   $ cd synthesizAR

and then use `pip` to install the needed dependencies,

.. code-block:: bash

   $ pip install .[all]

Additionally, if you'd like to run the test suite or build the documentation locally, you
can install the dev dependencies as follows,

.. code-block:: bash

   $ pip install .[test,docs,all]

To confirm that your installation is working, you can run the test suite,

.. code-block:: bash

   $ pytest synthesizAR

Updating
--------
As synthesizAR is not yet available on any package manager, the best way to keep up with releases is to pull down updates from GitHub. To grab the newest version from GitHub and install it, inside the package repo,

.. code-block:: bash

   $ git pull
   $ pip install .[all]

If you'd like to maintain a fork of synthesizAR (e.g. if you need to make changes to the codebase or contribute to the package), see the :doc:`Dev page </develop>`.
