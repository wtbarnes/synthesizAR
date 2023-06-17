"""
Useful decorators
"""
import functools

import astropy.units as u

__all__ = ['return_quantity_as_tuple']


def return_quantity_as_tuple(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        if isinstance(value, u.Quantity):
            return value.value, value.unit.to_string()
        else:
            return value
    return wrapper_decorator
