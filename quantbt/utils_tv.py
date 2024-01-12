import math

import numpy as np


def na(value):
    """
    Check if a value is considered as missing or not available.

    Parameters:
    - value: Any, the value to be checked.

    Returns:
    bool: True if the value is missing or not available, False otherwise.
    """
    if isinstance(value, float) and math.isnan(value):
        return True
    elif isinstance(value, np.ndarray) and np.isnan(value).any():
        return True
    elif value is None:
        return True
    return False


def ternary(condition, value_true, value_false):
    """
    Ternary operator implementation.

    Parameters:
    - condition: bool, the condition to check.
    - value_true: Any, the value to return if the condition is True.
    - value_false: Any, the value to return if the condition is False.

    Returns:
    Any: Either value_true or value_false based on the condition.
    """
    if condition:
        return value_true

    return value_false


def nz(value, replacement=0):
    """
    Replace missing or not available values with a specified replacement.

    Parameters:
    - value: Any, the value to be checked for being missing or not available.
    - replacement: Any, the value to be returned if 'value' is missing or not available. Default is 0.

    Returns:
    Any: Either the original 'value' or the specified 'replacement' based on whether 'value' is missing or not.

    Example:
    >>> nz(42)
    42

    >>> nz(None)
    0

    >>> nz(float('nan'), replacement=99)
    99
    """
    return ternary(na(value), replacement, value)
