# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.helper import enable_deprecations_as_exceptions
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
