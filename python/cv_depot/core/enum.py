from typing import Any  # noqa F401

from enum import Enum
import re

from lunchbox.enforce import Enforce
import numpy as np
# ------------------------------------------------------------------------------


'''
The enum module contains Enum classes for manging aspects of imagery such as bit
depths and video codecs.
'''


class EnumBase(Enum):
    def __repr__(self):
        # type: () -> str
        '''
        str: String representation of enum.
        '''
        return f'{self.__class__.__name__}.{self.name.upper()}'

    @classmethod
    def from_string(cls, string):
        # type: (str) -> EnumBase
        '''
        Constructs an enum instance from a given string.

        Args:
            string (int): Enum string.

        Raises:
            EnforceError: If value given is not a string.
            EnforceError: If no EnumBase type can be found for given string.

        Returns:
            EnumBase: Enum instance.
        '''
        msg = 'Value given is not a string. {a} != {b}.'
        Enforce(string, 'instance of', str, message=msg)

        lut = {x.name: x for x in cls.__members__.values()}
        string = string.upper().replace('-', '_')

        msg = '{a} is not a ' + cls.__name__ + ' option. '
        msg += f'Options: {sorted(lut.keys())}.'
        Enforce(string, 'in', lut.keys(), message=msg)

        return lut[string]


# BITDEPTH----------------------------------------------------------------------
class BitDepth(EnumBase):
    '''
    Legal bit depths.

    Includes:

        * FLOAT16
        * FLOAT32
        * UINT8
        * INT8
    '''
    FLOAT16 = (np.float16, 16, True, float)
    FLOAT32 = (np.float32, 32, True, float)
    UINT8 = (np.uint8, 8, False, int)
    INT8 = (np.int8, 8, True, int)

    def __init__(self, dtype, bits, signed, type_):
        # type: (Any, int, bool, type) -> None
        '''
        Args:
            dtype (numpy.type): Numpy datatype.
            bits (int): Number of bits per channel.
            signed (bool): Whether channel scalars are signed.
            type_ (type): Python type of scalar. Options include: [int, float].

        Returns:
            BitDepth: BitDepth instance.
        '''
        self.dtype = dtype  # type: ignore
        self.bits = bits
        self.signed = signed
        self.type_ = type_

    def __repr__(self):
        # type: () -> str
        return f'BitDepth.{self.name.upper()}'

    @staticmethod
    def from_dtype(dtype):
        # type: (Any) -> BitDepth
        '''
        Construct a BitDepth instance from a given numpy datatype.

        Args:
            dtype (numpy.type): Numpy datatype. Options include:
                [float16, float32, uint8, int8].

        Raises:
            TypeError: If invlaid dtype is given.

        Returns:
            BitDepth: BitDepth instance of given type.
        '''
        if dtype == np.float16:
            return BitDepth.FLOAT16
        elif dtype == np.float32:
            return BitDepth.FLOAT32
        elif dtype == np.uint8:
            return BitDepth.UINT8
        elif dtype == np.int8:
            return BitDepth.INT8

        # needed because of numpy malarkey with __name__
        if hasattr(dtype, '__name__'):
            dtype = dtype.__name__
        msg = f'{dtype} is not a supported bit depth.'
        raise TypeError(msg)


# IMAGE-------------------------------------------------------------------------
class ImageFormat(Enum):
    '''
    Legal image formats.

    Includes:

        * EXR
        * PNG
        * JPEG
        * TIFF
    '''
    EXR = (
        'exr', [BitDepth.FLOAT16, BitDepth.FLOAT32], list('rgba') + ['...'],
        1023, True
    )
    PNG = ('png', [BitDepth.UINT8], list('rgba'), 4, False)
    JPEG = ('jpeg', [BitDepth.UINT8], list('rgb'), 3, False)
    TIFF = (
        'tiff', [BitDepth.INT8, BitDepth.UINT8, BitDepth.FLOAT32],
        list('rgba') + ['...'], 500, False
    )

    def __init__(self, extension, bit_depths, channels, max_channels,
                 custom_metadata):
        # type: (str, list[BitDepth], list[str], int, bool) -> None
        '''
        Args:
            extension (str): Name of file extension.
            bit_depths (list[BitDepth]): Supported bit depths.
            channels (list[str]): Supported channels.
            max_channels (int): Maximum number of channels supported.
            custom_metadata (bool): Custom metadata support.

        Returns:
            ImageFormat: ImageFormat instance.
        '''
        self.extension = extension
        self.bit_depths = bit_depths
        self.channels = channels
        self.max_channels = max_channels
        self.custom_metadata = custom_metadata

    def __repr__(self):
        # type: () -> str
        return f'''
<ImageFormat.{self.name.upper()}>
      extension: {self.extension}
     bit_depths: {[x.name for x in self.bit_depths]}
       channels: {self.channels}
   max_channels: {self.max_channels}
custom_metadata: {self.custom_metadata}'''[1:]

    @staticmethod
    def from_extension(extension):
        '''
        Construct an ImageFormat instance for a given file extension.

        Args:
            extension (str): File extension.

        Raises:
            TypeError: If extension is illegal.

        Returns:
            ImageFormat: ImageFormat instance of given extension.
        '''
        exr_re = r'^\.?exr$'
        png_re = r'^\.?png$'
        jpeg_re = r'^\.?jpe?g$'
        tiff_re = r'^\.?tiff?$'

        if re.search(exr_re, extension, re.I):
            return ImageFormat.EXR

        elif re.search(png_re, extension, re.I):
            return ImageFormat.PNG

        elif re.search(jpeg_re, extension, re.I):
            return ImageFormat.JPEG

        elif re.search(tiff_re, extension, re.I):
            return ImageFormat.TIFF

        msg = f'ImageFormat not found for given extension: {extension}'
        raise TypeError(msg)


# VIDEO-------------------------------------------------------------------------
class VideoFormat(Enum):
    '''
    Legal video formats.

    Includes:

        * MP4
        * MPEG
        * MOV
        * M4V
    '''
    MP4 = ('mp4', [BitDepth.UINT8], list('rgb'), 3, False)
    MPEG = ('mpeg', [BitDepth.UINT8], list('rgb'), 3, False)
    MOV = ('mov', [BitDepth.UINT8], list('rgb'), 3, False)
    M4V = ('m4v', [BitDepth.UINT8], list('rgb'), 3, False)

    def __init__(self, extension, bit_depths, channels, max_channels,
                 custom_metadata):
        # type: (str, list[BitDepth], list[str], int, bool) -> None
        '''
        Args:
            extension (str): Name of file extension.
            bit_depths (list[BitDepth]): Supported bit depths.
            channels (list[str]): Supported channels.
            max_channels (int): Maximum number of channels supported.
            custom_metadata (bool): Custom metadata support.

        Returns:
            VideoFormat: VideoFormat instance.
        '''
        self.extension = extension
        self.bit_depths = bit_depths
        self.channels = channels
        self.max_channels = max_channels
        self.custom_metadata = custom_metadata

    def __repr__(self):
        # type: () -> str
        return f'''
<VideoFormat.{self.name.upper()}>
      extension: {self.extension}
     bit_depths: {[x.name for x in self.bit_depths]}
       channels: {self.channels}
   max_channels: {self.max_channels}
custom_metadata: {self.custom_metadata}'''[1:]

    @staticmethod
    def from_extension(extension):
        '''
        Construct an VideoFormat instance for a given file extension.

        Args:
            extension (str): File extension.

        Raises:
            TypeError: If extension is invalid.

        Returns:
            VideoFormat: VideoFormat instance of given extension.
        '''
        mp4_re = r'^\.?mp4$'
        mpeg_re = r'^\.?mpe?g$'
        mov_re = r'^\.?mov$'
        m4v_re = r'^\.?m4v$'

        if re.search(mp4_re, extension, re.I):
            return VideoFormat.MP4

        if re.search(mpeg_re, extension, re.I):
            return VideoFormat.MPEG

        if re.search(mov_re, extension, re.I):
            return VideoFormat.MOV

        if re.search(m4v_re, extension, re.I):
            return VideoFormat.M4V

        msg = f'VideoFormat not found for given extension: {extension}'
        raise TypeError(msg)
# ------------------------------------------------------------------------------


class VideoCodec(Enum):
    '''
    Legal video codecs.

    Includes:

        * H264
        * H265
    '''
    H264 = ('h264', 'h264')
    H265 = ('h265', 'hevc')

    def __init__(self, string, ffmpeg_code):
        # type: (str, str) -> None
        '''
        Args:
            string (str): String representation of codec.
            ffmpeg_code (str): FFMPEG code.
        '''
        self.string = string
        self.ffmpeg_code = ffmpeg_code

    def __repr__(self):
        # type: () -> str
        return f'''
<VideoCodec.{self.name.upper()}>
  string: {self.string}
  ffmpeg_code: {self.ffmpeg_code}'''[1:]


# DIRECTION---------------------------------------------------------------------
class Direction(EnumBase):
    '''
    Legal directions.

    Includes:

        * TOP
        * BOTTOM
        * LEFT
        * RIGHT
    '''
    TOP = ('top')
    BOTTOM = ('bottom')
    LEFT = ('left')
    RIGHT = ('right')


class Anchor(EnumBase):
    '''
    Legal anchors.

    Includes:

        * TOP_LEFT
        * TOP_CENTER
        * TOP_RIGHT
        * CENTER_LEFT
        * CENTER_CENTER
        * CENTER_RIGHT
        * BOTTOM_LEFT
        * BOTTOM_CENTER
        * BOTTOM_RIGHT
    '''
    TOP_LEFT = ('top', 'left')
    TOP_CENTER = ('top', 'center')
    TOP_RIGHT = ('top', 'right')
    CENTER_LEFT = ('center', 'left')
    CENTER_CENTER = ('center', 'center')
    CENTER_RIGHT = ('center', 'right')
    BOTTOM_LEFT = ('bottom', 'left')
    BOTTOM_CENTER = ('bottom', 'center')
    BOTTOM_RIGHT = ('bottom', 'right')
