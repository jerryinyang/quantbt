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