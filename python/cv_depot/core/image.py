from typing import Any, Optional, Tuple, Union  # noqa F401
from numpy.typing import NDArray  # noqa F401
from cv_depot.core.types import Filepath  # noqa F401

from copy import deepcopy
from itertools import combinations, chain
from pathlib import Path
import os
import re

from lunchbox.enforce import Enforce
from openexr_tools.enum import ImageCodec
import numpy as np
import openexr_tools.tools as exrtools
import PIL.Image as pil

from cv_depot.core.enum import BitDepth, ImageFormat
from cv_depot.core.viewer import ImageViewer
import cv_depot.core.tools as cvt
# ------------------------------------------------------------------------------


def _has_super_darks(image):
    # type: (Image) -> bool
    '''
    Determines if given image has values below 0.0

    Args:
        image (Image): Image instance.

    Raises:
        EnforceError: If image is not an Image instance.

    Returns:
        bool: Presence of super darks.
    '''
    Enforce(image, 'instance of', Image)
    return bool(image.data.min() < 0.0)


def _has_super_brights(image):
    # type: (Image) -> bool
    '''
    Determines if given image has values above 1.0

    Args:
        image (Image): Image instance.

    Raises:
        EnforceError: If image is not an Image instance.

    Returns:
        bool: Presence of super brights.
    '''
    Enforce(image, 'instance of', Image)
    return bool(image.data.max() > 1.0)


class Image():
    '''
    Class for reading, writing, converting and displaying properties of images.
    '''
    @staticmethod
    def from_array(array):
        # type: (NDArray) -> Image
        '''
        Construct an Image instance from a given numpy array.

        Args:
            array (numpy.NDArray): Numpy array.

        Returns:
            Image: Image instance of given numpy array.
        '''
        # enforce bit depth compliance
        BitDepth.from_dtype(array.dtype)
        return Image(array.copy(), {}, None, allow=True)

    @staticmethod
    def from_pil(image):
        # type: (pil.Image) -> Image
        '''
        Construct an Image instance from a given PIL Image.

        Args:
            image (pil.Image): PIL Image.

        Returns:
            Image: Image instance of a given PIL Image.
        '''
        return Image.from_array(np.array(image))

    @staticmethod
    def read(filepath):
        # type: (Filepath) -> Image
        '''
        Constructs an Image instance given a full path to an image file.

        Args:
            filepath (str or Path): Image filepath.

        Raises:
            FileNotFoundError: If file could not be found on disk.
            TypeError: If filepath is not a str or Path.

        Returns:
            Image: Image instance of given file.
        '''
        metadata = {}  # type: dict[str, Any]
        format_ = None

        if isinstance(filepath, Path):
            filepath = filepath.absolute().as_posix()

        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                msg = f'{filepath} does not exist.'
                raise FileNotFoundError(msg)

            _, ext = os.path.splitext(filepath)
            format_ = ImageFormat.from_extension(ext)

            if format_ is ImageFormat.EXR:
                data, metadata = exrtools.read_exr(filepath)
            else:
                data = np.asarray(pil.open(filepath))

        else:
            msg = f'Object of type {filepath.__class__.__name__} '
            msg += 'is not a str or Path.'
            raise TypeError(msg)

        return Image(data, metadata, format_, allow=True)

    def __init__(self, data, metadata={}, format_=None, allow=False):
        # type: (NDArray, dict[str, Any], Optional[ImageFormat], bool) -> None
        '''
        This constructor should not be called directly except internally and in
        testing.

        Args:
            data (numpy.NDArray): Image.
            metadata (dict, optional): Image metadata. Default: {}.
            format_ (ImageFormat, optional): Format of image. Default: None.
            allow (bool, optional): Whether to allow construction using init.
                Default: False.

        Raises:
            AttributeError: If image data dimensions are not 2 or 3.

        Returns:
            Image: Image instance.
        '''
        if not allow:
            msg = "Please call one of Image's static constructors to create an "
            msg += 'instance. Options include: read, from_array.'
            raise NotImplementedError(msg)

        # ensure data has 3 dimensions
        shape = data.shape
        dims = len(shape)
        if dims > 3 or dims < 2:
            msg = f'Illegal number of dimensions for image data. {dims} not in '
            msg += '[2, 3].'
            raise AttributeError(msg)

        if dims == 2:
            data = data.reshape((*shape, 1))

        self._data = data
        self.metadata = metadata
        self.format = format_
    # --------------------------------------------------------------------------

    def _repr(self):
        # type: () -> str
        fmat = str(None)
        if self.format is not None:
            fmat = self.format.name

        return f'''
       WIDTH: {self.width}
      HEIGHT: {self.height}
NUM_CHANNELS: {self.num_channels}
   BIT_DEPTH: {self.bit_depth.name}
      FORMAT: {fmat}'''[1:]

    def _repr_html_(self):
        # type: () -> None
        '''
        Creates a HTML representation of image data.
        '''
        ImageViewer(self).show()

    def _repr_png(self):
        # type: () -> Optional[bytes]
        '''
        Creates a PNG representation of image data.

        Returns:
            str: PNG.
        '''
        this = self
        if _has_super_brights(self) or _has_super_darks(self):
            this = self.to_unit_space()
        output = this.to_bit_depth(BitDepth.UINT8).to_pil()._repr_png_()
        return output

    def _string_to_channels(self, string):
        # type: (str) -> list
        '''
        Converts string to list of channels.

        Args:
            string (str): String representation of channels.

        Returns:
            list: List of channels.
        '''
        # special rgba short circuit
        combos = [list(combinations('rgba', i)) for i in range(1, 5)]  # type: Any
        combos = list(map(set, chain(*combos)))
        if set(string) in combos:
            return list(string)

        # if channels is actually a layer name
        if string in self.channel_layers:
            found = list(filter(
                lambda x: re.search(string + r'\..+', str(x)),
                self.channels
            ))
            if found != []:
                # found channels that matched [layer-name].[channel] pattern
                return found
            return list(string)
        return [string]

    def __getitem__(self, indices):
        # type: (Union[int, tuple, list, slice, str]) -> Image
        '''
        Gets slice of image data. Indices are given in the order:
        width, height, channel.

        Args:
            indices (int, tuple, list, slice, str): Slice of image data.

        Raises:
            IndexError: If number of indices provided is greater than 3.
            IndexError: If channel given is illegal.
            IndexError: If three lists are given as indices.

        Returns:
            Image: Image slice.
        '''
        if not isinstance(indices, tuple) or isinstance(indices, list):
            indices = [indices]

        size = len(indices)
        if size > 3:
            msg = f'Number of dimensions provided: {size}, is greater than 3.'
            raise IndexError(msg)

        # convert indices to triplet of slices
        columns = slice(None, None)  # type: Any
        rows = slice(None, None)  # type: Any
        channels = slice(None, None)  # type: Any
        if size == 3:
            columns, rows, channels = indices
        elif size == 2:
            columns, rows = indices
        else:
            columns = indices[0]

        # convert channels to list of indices
        channel_meta = self.metadata.get('channels', [])
        if channels.__class__.__name__ in ['str', 'tuple', 'list']:
            if isinstance(channels, str):
                channels = self._string_to_channels(channels)
            channel_meta = channels
            chans = []
            for channel in channels:
                if isinstance(channel, str):
                    if channel not in self.channels:
                        msg = f'{channel} is not a legal channel name.'
                        raise IndexError(msg)
                    channel = self.channels.index(channel)
                chans.append(channel)

            if len(chans) == 1:
                chans = chans[0]
            channels = chans

        # coerce to list for simpler logic
        if isinstance(columns, tuple):
            columns = list(columns)
        if isinstance(rows, tuple):
            rows = list(rows)

        types = [
            columns.__class__.__name__,
            rows.__class__.__name__,
            channels.__class__.__name__,
        ]
        if types == ['list', 'list', 'list']:
            msg = 'Three lists are not acceptable as indices.'
            raise IndexError(msg)

        if isinstance(channels, slice):
            channel_meta = self.channels[channels]

        data = self._data[rows, columns, channels]
        metadata = deepcopy(self.metadata)
        metadata['channels'] = channel_meta
        return Image(data, metadata=metadata, format_=self.format, allow=True)
    # --------------------------------------------------------------------------

    def set_channels(self, channels):
        # type: (list[Union[str, int]]) -> Image
        '''
        Set's channels names.

        Args:
            channels (list[str or int]): List of channel names:

        Raises:
            ValueError: If number of channels given doesn't not equal data
                shape.
            ValueError: If duplicate channel names found.

        Returns:
            Image: self.
        '''
        if len(channels) != self.num_channels:
            msg = 'Number of channels given does not equal last dimension size.'
            msg += f' {len(channels)} != {self.num_channels}.'
            raise ValueError(msg)

        uniq = set(channels)
        if len(uniq) < len(channels):
            for c in uniq:
                channels.remove(c)
            msg = f'Duplicate channel names found: {channels}.'
            raise ValueError(msg)

        metadata = deepcopy(self.metadata)
        metadata['channels'] = channels
        return Image(
            self._data.copy(),
            metadata=metadata,
            format_=self.format,
            allow=True,
        )

    def write(self, filepath, codec=ImageCodec.PIZ):
        # type: (Filepath, ImageCodec) -> None
        '''
        Write image to file.

        Args:
            filepath (str or Path): Full path to image file.
            codec (ImageCodec, optional): EXR compression scheme to be used.
                Default: ImageCodec.PIZ.

        Raises:
            TypeError: If format does not support instance bit depth.
            AttributeError: If format does not support the number of channels in
                instance.
        '''
        if isinstance(filepath, Path):
            filepath = filepath.absolute().as_posix()

        _, ext_ = os.path.splitext(filepath)
        ext = ImageFormat.from_extension(ext_)

        # ensure format is compatible with image data
        if self.bit_depth not in ext.bit_depths:
            msg = f'{ext.name} cannot be written with {self.bit_depth.name}'
            msg += ' data.'
            raise TypeError(msg)

        if self.num_channels > ext.max_channels:
            msg = f'{ext.name} cannot be written with {self.num_channels} '
            msg += f'channels. Max channels supported: {ext.max_channels}.'
            raise AttributeError(msg)

        # write data
        if ext is ImageFormat.EXR:
            metadata = self.metadata
            metadata['channels'] = self.channels
            exrtools.write_exr(filepath, self._data, metadata, codec)

        else:
            pil.fromarray(self._data).save(filepath, format=ext.name)

    def to_bit_depth(self, bit_depth):
        # type: (BitDepth) -> Image
        '''
        Convert image to given bit depth.
        Warning: Numpy's conversions for INT8 are bizarre.

        Args:
            bit_depth (BitDepth): Target bit depth.

        Raises:
            ValueError: If converting from float to 8-bit and values exceed 1.
            ValueError: If converting from float to 8-bit and values less than 0.

        Returns:
            Image: New Image instance at given bit depth.
        '''
        image = self._data
        src = self.bit_depth
        tgt = bit_depth

        if src is tgt:
            return self

        elif src is BitDepth.UINT8 and tgt.type_ is float:
            image = image.astype(tgt.dtype) / 255

        elif src.type_ is float and tgt.bits == 8:
            if _has_super_darks(self):
                msg = f'Image has values below 0. Min value: {image.min()}'
                raise ValueError(msg)

            if _has_super_brights(self):
                msg = f'Image has values above 1. Max value: {image.max()}'
                raise ValueError(msg)

            image = (image * 255).astype(tgt.dtype)

        else:
            image = image.astype(tgt.dtype)

        metadata = deepcopy(self.metadata)
        return Image(image, metadata=metadata, format_=self.format, allow=True)

    def to_unit_space(self):
        # type: () -> Image
        '''
        Normalizes image to [0, 1] range.

        Returns:
            Image: Normalized image.
        '''
        data = self.to_bit_depth(BitDepth.FLOAT32)._data.copy()
        max_, min_ = data.max(), data.min()
        data = (data - min_) / (max_ - min_)
        data = Image.from_array(data).to_bit_depth(self.bit_depth)._data
        metadata = deepcopy(self.metadata)
        return Image(data, metadata=metadata, format_=self.format, allow=True)

    def to_array(self):
        # type: () -> NDArray
        '''
        Returns numpy array.

        Returns:
            numpy.NDArray: Image as numpy array.
        '''
        return self.data

    def to_pil(self):
        # type: () -> pil.Image
        '''
        Returns pil.Image.

        Returns:
            pil: Image as pil.Image.
        '''
        if self.num_channels == 1:
            mode = 'L'
        elif self.num_channels == 3:
            mode = 'RGB'
        elif self.num_channels == 4:
            mode = 'RGBA'
        else:
            raise ValueError('PIL only accepts image with 1, 3 or 4 channels.')
        return pil.fromarray(self.data, mode=mode)

    def compare(self, image, content=False, diff_only=False):
        # type: (Image, bool, bool) -> dict[str, Any]
        '''
        Compare this image with a given image.

        Args:
            image (Image): Image to compare.
            content (bool, optional): If True, compare data. Default: False.
            diff_only (bool, optional): If True, only return the keys with
                differing values. Default: False.

        Raises:
            EnforceError: If image is not an Image instance.
            ValueError: IF content is True and images cannot be compared.

        Returns:
            dict: A dictionary of comparisons.
        '''
        msg = 'Image must be an instance of Image.'
        Enforce(image, 'instance of', Image, message=msg)
        # ----------------------------------------------------------------------

        a = self.info
        b = image.info
        output = {}  # type: dict[str, Any]
        for k, v in a.items():
            output[k] = (v, b.get(k, None))
        for k, v in b.items():
            if k not in output.keys():
                output[k] = (a.get(k, None), v)

        if diff_only:
            for k, v in list(output.items()):
                if v[0] == v[1]:
                    del output[k]

        if content:
            x = self.to_bit_depth(BitDepth.FLOAT16).data
            y = image.to_bit_depth(BitDepth.FLOAT16).data

            try:
                diff = float(abs(x - y).mean())
            except ValueError as e:
                raise ValueError(f'Cannot compare images: {e}')

            output['mean_content_difference'] = diff
            if diff_only and diff == 0:
                del output['mean_content_difference']

        return output

    def __eq__(self, image):
        # type: (object) -> bool
        '''
        Compare this image with a given image.

        Returns:
            bool: True if images are equal.
        '''
        return self.compare(image, content=True, diff_only=True) == {}  # type: ignore
    # --------------------------------------------------------------------------

    @property
    def data(self):
        # type: () -> NDArray
        '''
        numpy.NDArray: Image data.
        '''
        if self.num_channels == 1:
            return np.squeeze(self._data, axis=2)
        return self._data

    @property
    def info(self):
        # type: () -> dict[str, Any]
        '''
        dict: A dictionary of all information about the Image instance.
        '''
        output = dict(
            width=self.width,
            height=self.height,
            channels=self.channels,
            num_channels=self.num_channels,
            bit_depth=self.bit_depth.name,
            dtype=self.bit_depth.dtype,
            bits=self.bit_depth.bits,
            signed=self.bit_depth.signed,
            type=self.bit_depth.type_,
            format_extension=None,
            format_bit_depths=None,
            format_channels=None,
            format_max_channels=None,
            format_custom_metadata=None,
        )
        if self.format is not None:
            fmat = dict(
                format_extension=self.extension,
                format_bit_depths=self.format.bit_depths,
                format_channels=self.format.channels,
                format_max_channels=self.format.max_channels,
                format_custom_metadata=self.format.custom_metadata,
            )
            output.update(fmat)
        return output

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int]
        '''
        tuple[int]: (width, height, channels) of image.
        '''
        return (self.width, self.height, self.num_channels)

    @property
    def width(self):
        # type: () -> int
        '''
        int: Width of image.
        '''
        return self._data.shape[1]

    @property
    def height(self):
        # type: () -> int
        '''
        int: Height of image.
        '''
        return self._data.shape[0]

    @property
    def width_and_height(self):
        # type: () -> Tuple[int, int]
        '''
        tupe[int]: (width, height) of image.
        '''
        return (self.width, self.height)

    @property
    def channels(self):
        # type: () -> list[Union[str, int]]
        '''
        list[str or int]: List of channel names.
        '''
        if 'channels' in self.metadata:
            return self.metadata['channels']
        return cvt.get_channels_from_array(self._data)

    @property
    def num_channels(self):
        # type: () -> int
        '''
        int: Number of channels in image.
        '''
        return len(self.channels)

    @property
    def max_channels(self):
        # type: () -> Optional[int]
        '''
        int: Maximum number of channels supported by image format.
        '''
        if self.format is None:
            return None
        return self.format.max_channels

    @property
    def channel_layers(self):
        # type: () -> list[str]
        '''
        list[str]: List of channel layers.
        '''
        channels = [str(x) for x in self.channels]
        with_layer = list(filter(lambda x: '.' in str(x), channels))  # type: list[str]
        wo_layer_name = list(filter(lambda x: '.' not in str(x), channels))

        # break out channels without layer names into groups of 4
        len_ = len(wo_layer_name)
        layers = []
        for idx in range(0, len_, 4):
            layer_ = wo_layer_name[idx:min(idx + 4, len_)]
            layer = ''.join(layer_)
            layers.append(layer)

        # append all unique layers of channels with layer names
        for chan in with_layer:
            layer = ''.join(chan.split('.')[0])
            if layer not in layers:
                layers.append(layer)
        return layers

    @property
    def bit_depth(self):
        # type: () -> BitDepth
        '''
        BitDepth: Bit depth of image.
        '''
        return BitDepth.from_dtype(self._data.dtype)

    @property
    def extension(self):
        # type: () -> Optional[str]
        '''
        str: Image format extension.
        '''
        if self.format is None:
            return None
        return self.format.extension
