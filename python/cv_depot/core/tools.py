from typing import Union  # noqa F401
from numpy.typing import NDArray  # noqa F401

from pandas import Series
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# ------------------------------------------------------------------------------


def get_channels_from_array(array):
    # type: (NDArray) -> list[Union[str, int]]
    '''
    Returns a list of strings representing the given array's channels.
    If array has only one channel then ['l'] is returned.
    First 4 channels are [r, g, b, a], in that order. All subsequent channels
    are integers starting at 4.

    Args:
        array (numpy.NDArray): Numpy array with 2+ dimensional shape.

    Returns:
        list[str and int]: Channels.
    '''
    if len(array.shape) < 3 or array.shape[2] == 1:
        return ['l']
    else:
        temp = list(range(array.shape[2]))
        lut = {0: 'r', 1: 'g', 2: 'b', 3: 'a'}  # type: dict[int, str]
        channels = []  # type: list[Union[str, int]]
        for i in temp:
            c = i  # type: Union[int, str]
            if i in lut:
                c = lut[i]
            channels.append(c)
        return channels


def apply_minmax(item, floor=-10**16, ceiling=10**16):
    # type: (Series, float, float) -> Series
    '''
    Normalizes data in given Series according to it minimum and maximum values.

    Args:
        item (Series): A pandas Series object full of numbers.
        floor (float, optional): The value NaN and -inf should be converted to.
        ceiling (float, optional): The value inf should be converted to.

    Returns:
        Series: Normalized Series.
    '''
    if floor > ceiling:
        msg = f'Floor must not be greater than ceiling. {floor} > {ceiling}.'
        raise ValueError(msg)

    output = item\
        .apply(lambda x: floor if np.isnan(x) else x)\
        .apply(lambda x: floor if np.isinf(x) and x < 0 else x)\
        .apply(lambda x: ceiling if np.isinf(x) and x > 0 else x)
    output = np.array(output.tolist()).reshape(-1, 1)
    output = MinMaxScaler().fit_transform(output)
    output = np.squeeze(output).tolist()
    output = Series(output)
    return output

