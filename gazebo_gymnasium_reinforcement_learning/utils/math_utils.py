import numpy as np


def normalize_to_range(targeted_value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param targeted_value: value to be normalized [-1,1] likely
    :param min_val: value's min value
    :param max_val: value's max value
    :param new_min: normalized range min value
    :param new_max: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value in range [new_min, new_max]
    """
    targeted_value = float(targeted_value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (targeted_value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (targeted_value - max_val) + new_max
