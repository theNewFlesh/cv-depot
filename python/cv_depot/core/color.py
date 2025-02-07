from typing import Any, Optional, Union  # noqa F401
from numpy.typing import NDArray  # noqa F401

import json
import math
from copy import copy
from enum import Enum

from lunchbox.enforce import Enforce
import numpy as np

from cv_depot.core.enum import BitDepth
# ------------------------------------------------------------------------------


'''
Color is a fundamentally a vector of length c, where c is number of channels.
Colors can only be of bit depths supported by the BitDepth class.
'''


class BasicColor(Enum):
    '''
    A convenience enum for basic, common color vectors.
    Legal colors include:

        * BLACK
        * WHITE
        * GREY
        * RED
        * GREEN
        * BLUE
        * YELLOW
        * MAGENTA
        * CYAN
    '''
    BLACK = ('#000000', [0.0], [0.0, 0.0, 0.0])
    WHITE = ('#FFFFFF', [1.0], [1.0, 1.0, 1.0])
    GREY = ('#808080', [0.5], [0.5, 0.5, 0.5])
    RED = ('#FF0000', None, [1.0, 0.0, 0.0])
    GREEN = ('#00FF00', None, [0.0, 1.0, 0.0])
    BLUE = ('#0000FF', None, [0.0, 0.0, 1.0])
    YELLOW = ('#FFFF00', None, [1.0, 1.0, 0.0])
    MAGENTA = ('#FF00FF', None, [1.0, 0.0, 1.0])
    CYAN = ('#00FFFF', None, [0.0, 1.0, 1.0])

    # henanigans
    BG = ('#242424', [0.141], [0.141, 0.141, 0.141])
    BLUE1 = ('#5F95DE', None, [0.373, 0.584, 0.871])
    BLUE2 = ('#93B6E6', None, [0.576, 0.714, 0.902])
    CYAN1 = ('#7EC4CF', None, [0.494, 0.769, 0.812])
    CYAN2 = ('#B6ECF3', None, [0.714, 0.925, 0.953])
    DARK1 = ('#040404', [0.016], [0.016, 0.016, 0.016])
    DARK2 = ('#141414', [0.078], [0.078, 0.078, 0.078])
    DIALOG1 = ('#444459', None, [0.267, 0.267, 0.349])
    DIALOG2 = ('#5D5D7A', None, [0.365, 0.365, 0.478])
    GREEN1 = ('#8BD155', None, [0.545, 0.82, 0.333])
    GREEN2 = ('#A0D17B', None, [0.627, 0.82, 0.482])
    GREY1 = ('#343434', [0.204], [0.204, 0.204, 0.204])
    GREY2 = ('#444444', [0.267], [0.267, 0.267, 0.267])
    LIGHT1 = ('#A4A4A4', [0.643], [0.643, 0.643, 0.643])
    LIGHT2 = ('#F4F4F4', [0.957], [0.957, 0.957, 0.957])
    ORANGE1 = ('#EB9E58', None, [0.922, 0.62, 0.345])
    ORANGE2 = ('#EBB483', None, [0.922, 0.706, 0.514])
    PURPLE1 = ('#C98FDE', None, [0.788, 0.561, 0.871])
    PURPLE2 = ('#AC92DE', None, [0.675, 0.573, 0.871])
    RED1 = ('#F77E70', None, [0.969, 0.494, 0.439])
    RED2 = ('#DE958E', None, [0.871, 0.584, 0.557])
    YELLOW1 = ('#E8EA7E', None, [0.91, 0.918, 0.494])
    YELLOW2 = ('#E9EABE', None, [0.914, 0.918, 0.745])

    def __init__(self, hexidecimal, one_channel, three_channel):
        # type: (str, list[float], list[float]) -> None
        '''
        Args:
            hexidecimal (str): Hexidecimal representation of color.
            one_channel (list[float]): List with single float.
            three_channel (list[float]): List with three floats.

        Returns:
            BasicColor: BasicColor instance.
        '''
        self._hexidecimal = hexidecimal
        self._one_channel = one_channel
        self._three_channel = three_channel

    @staticmethod
    def _get_color(
        value,  # type: Union[str, BasicColor, list[int], list[float]]
        attr    # type: str
    ):          # type: (...) -> BasicColor
        '''
        Finds BasicColor instance given a value and an attribute name.

        Args:
            value (str or BasicColor): BasicColor value to be looked up.
            attr (str): Attribute name to be used in lookup.

        Raises:
            ValueError: If no BasicColor corresponds to given value.

        Returns:
            BasicColor: BasicColor instance.
        '''
        lut = {
            json.dumps(getattr(x, attr)): x for x in BasicColor
        }  # type: dict[str, BasicColor]

        if attr == 'string':
            value = str(value).lower()
        elif attr == 'hexidecimal':
            value = str(value).upper()

        key = json.dumps(value)
        if key not in lut.keys():
            msg = f'{value} is not a legal color.'
            raise ValueError(msg)
        return lut[key]

    @staticmethod
    def from_string(value):
        # type: (str) -> BasicColor
        '''
        Contructs a BasicColor instance from a given string.

        Args:
            value (str): Name of color.

        Returns:
            BasicColor: BasicColor instance.
        '''
        return BasicColor._get_color(value, 'string')

    @staticmethod
    def from_list(value):
        # type: (list) -> BasicColor
        '''
        Contructs a BasicColor instance from a given list of floats or ints.

        Args:
            value (list): BasicColor as list of numbers.

        Returns:
            BasicColor: BasicColor instance.
        '''
        value = list(map(float, value))
        if len(value) == 1:
            return BasicColor._get_color(value, 'one_channel')
        elif len(value) == 3:
            return BasicColor._get_color(value, 'three_channel')

        msg = f'Invalid color value {value}. Must be 1 or 3 channels.'
        raise ValueError(msg)

    @staticmethod
    def from_list_8_bit(value):
        # type: (list[int]) -> BasicColor
        '''
        Contructs a BasicColor instance from a given list of ints between 0 and
        255.

        Args:
            value (list): BasicColor as list of numbers.

        Returns:
            BasicColor: BasicColor instance.
        '''
        value_ = [x / 255 for x in value]
        return BasicColor.from_list(value_)

    @staticmethod
    def from_hexidecimal(value):
        # type: (str) -> BasicColor
        '''
        Contructs a BasicColor instance from a given hexidecimal string.

        Args:
            value (str): Hexidecimal value of color.

        Returns:
            BasicColor: BasicColor instance.
        '''
        return BasicColor._get_color(value, 'hexidecimal')

    def __repr__(self):
        # type: () -> str
        return f'''
<BasicColor.{self.name.upper()}>
             string: {self.string}
        hexidecimal: {self.hexidecimal}
        one_channel: {self.one_channel}
      three_channel: {self.three_channel}
  one_channel_8_bit: {self.one_channel_8_bit}
three_channel_8_bit: {self.three_channel_8_bit}'''[1:]

    @property
    def hexidecimal(self):
        # type: () -> str
        '''
        str: Hexidecimal representation of color.
        '''
        return copy(self._hexidecimal)

    @property
    def one_channel(self):
        # type: () -> list[float]
        '''
        list[float]: One channel, floating point representation of color.
        '''
        return copy(self._one_channel)

    @property
    def three_channel(self):
        # type: () -> list[float]
        '''
        list[float]: Three channel, floating point representation of color.
        '''
        return copy(self._three_channel)

    @property
    def one_channel_8_bit(self):
        # type: () -> list[int]
        '''
        list[int]: One channel, 8 bit representation of color.
        '''
        if self.one_channel is None:
            return None
        return [int(math.ceil(self.one_channel[0] * 255))]

    @property
    def three_channel_8_bit(self):
        # type: () -> list[int]
        '''
        list[int]: Three channel, 8 bit representation of color.
        '''
        return [int(math.ceil(x * 255)) for x in self.three_channel]

    @property
    def string(self):
        # type: () -> str
        '''
        string: String representation of color.
        '''
        return self.name.lower()
# ------------------------------------------------------------------------------


class Color:
    '''
    Makes working with color vectors easy.
    '''
    bit_depth = BitDepth.FLOAT32

    def __init__(self, data):
        # type: (NDArray) -> None
        '''
        Constructs a Color instance.

        Args:
            data (numpy.NDArray): A 1 dimensional numpy array of shape (n,).

        Raises:
            TypeError: If data is not a numpy array.
            AttributeError: If data's number of dimensions is not 1.

        Returns:
            Color: Color instance.
        '''
        if not isinstance(data, np.ndarray):
            msg = 'Data must be a numpy array.'
            raise TypeError(msg)

        if data.ndim != 1:
            msg = 'Arrays must be one dimensional, so that its shape is (n,). '
            msg += f'Given array has {data.ndim} dimensions and a shape of '
            msg += f'{data.shape}.'
            raise AttributeError(msg)

        # checks for legal bit depth
        bit_depth = BitDepth.from_dtype(data.dtype)
        data = data.astype(self.bit_depth.dtype)

        # 8 bit conversion
        if bit_depth is BitDepth.INT8:
            data = (data + 127) / 255
        elif bit_depth is BitDepth.UINT8:
            data = data / 255

        self._data = data

    @staticmethod
    def from_array(data, num_channels=None, fill_value=0):
        # type: (NDArray, Optional[int], float) -> Color
        '''
        Contructs a Color instance from a given numpy array.

        Args:
            data (numpy.NDArray): Numpy array.
            num_channels (int, optional): Number of desired channels in the
                Color instance. Number of channels equals len(data) if set to
                None. Default: None.
            fill_value (float, optional): Value used to fill additional
                channels. Default: 0.

        Returns:
            Color: Color instance of given data.
        '''
        if num_channels is not None:
            Enforce(num_channels, 'instance of', int)
            Enforce(num_channels, '>=', 1)

            diff = num_channels - len(data)
            if diff > 0:
                buff = np.ones(diff, data.dtype) * fill_value
                data = np.append(data, buff)
            data = data[:num_channels]
        return Color(data)

    @staticmethod
    def from_list(
        data, num_channels=None, fill_value=0, bit_depth=BitDepth.FLOAT32
    ):
        # type: (NDArray, Optional[int], float, BitDepth) -> Color
        '''
        Contructs a Color instance from a given list of ints or floats.

        Args:
            data (list): List of ints or floats.
            num_channels (int, optional): Number of desired channels in the
                Color instance. Number of channels equals len(data) if set to
                None. Default: None.
            fill_value (float, optional): Value used to fill additional
                channels. Default: 0.
            bit_depth (BitDepth, optional): Bit depth of given list.
                Default: BitDepth.FLOAT32.

        Returns:
            Color: Color instance of given data.
        '''
        data = np.array(data, dtype=bit_depth.dtype)
        return Color.from_array(
            data, num_channels=num_channels, fill_value=fill_value
        )

    @staticmethod
    def from_basic_color(data, num_channels=3, fill_value=0.0):
        # type: (BasicColor, int, float) -> Color
        '''
        Contructs a Color instance from a BasicColor enum.

        Args:
            data (BasicColor): BasicColor enum.
            num_channels (int, optional): Number of desired channels in the
                Color instance. Default: 3.
            fill_value (float, optional): Value used to fill additional
                channels. Default: 0.

        Raises:
            EnforceError: If num_channels is less than 1.
            EnforceError: If no shape is one channel and no one channel
                equivalent of the given color could be found.

        Returns:
            Color: Color instance of given data.
        '''
        msg = 'num_channels must be greater than or equal to 1. {a} < {b}.'
        Enforce(num_channels, '>=', 1, message=msg)

        output = data  # type: Any
        if num_channels == 1:
            msg = f'No one channel equivalent found for given color: {data}.'
            Enforce(data.one_channel, '!=', None, message=msg)
            output = data.one_channel
        else:
            output = data.three_channel

        return Color.from_list(
            output,
            num_channels=num_channels,
            fill_value=fill_value,
            bit_depth=BitDepth.FLOAT32
        )

    def __repr__(self):
        # type: () -> str
        '''
        Represention of Color instance includes:

            * values - actual numpy array
            * bit_depth - always FLOAT32
            * num_channels - number of color channels
            * name - displays only if the color values corresponds to a
              BasicColor
        '''
        output = [
            '<Color>',
            f'      values: {self._data}',
            f'   bit_depth: {self.bit_depth.name.upper()}',
            f'num_channels: {self.num_channels}',
        ]  # type: Any

        try:
            items = self.to_array().tolist()
            name = BasicColor.from_list(items).string  # type: ignore
            output.append(f'        name: {name}')
        except Exception:
            pass

        output = '\n'.join(output)
        return output

    @property
    def num_channels(self):
        # type: () -> int
        '''
        Number of channels in color vector.
        '''
        return len(self._data)

    def to_array(self, bit_depth=BitDepth.FLOAT32):
        # type: (BitDepth) -> NDArray
        '''
        Returns color as a numpy array at a given bit depth.

        Args:
            bit_depth (BitDepth, optional): Bit depth of output.
                Default: BitDepth.FLOAT32.

        Returns:
            numpy.NDArray: Color vector as numpy array.
        '''
        output = self._data
        if bit_depth is BitDepth.INT8:
            output = np.ceil(output * 255) - 128
        elif bit_depth is BitDepth.UINT8:
            output = np.ceil(output * 255)
        return output.astype(bit_depth.dtype)
