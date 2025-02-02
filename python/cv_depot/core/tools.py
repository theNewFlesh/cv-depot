from typing import Union  # noqa F401
from numpy.typing import NDArray  # noqa F401
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
