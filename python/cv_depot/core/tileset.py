from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # noqa F401
from cv_depot.core.color import Color  # noqa F401

from pathlib import Path
import math
import os
import re

from hidebound.core.specification_base import SpecificationBase
from lunchbox.enforce import Enforce, EnforceError
from openexr_tools.enum import ImageCodec
from pandas import DataFrame
import hidebound.core.tools as hbt
import pandas as pd

from cv_depot.core.color import BasicColor
from cv_depot.core.enum import BitDepth
from cv_depot.core.image import Image
from vision.pipeline.specifications import Tset001
import cv_depot.ops as ops
import vision.enforce.enforce_tools as eft
import vision.pipeline.pipeline_tools as plt
import vision.utils as utils
# ------------------------------------------------------------------------------


class Tileset:
    '''
    The Tileset class is used for converting large images to and from tiled
    image data. It represents tiles internally as a pandas DataFrame with the
    main columns:

        * width - width coordinate (x)
        * height - height coordinate (y)
        * depth - depth coordinate (z)
        * tile_width - width of each tile in pixels
        * tile_height - height of each tile in pixels
        * num_channels - number of channels per tile
        * bit_depth - bit depth of each tile
        * content - Image instances containing tile data
    '''

    SPECIFICATION = Tset001  # type: SpecificationBase
    SAMPLE_SIZE = 10  # type: int

    def __init__(self, data):
        # type: (DataFrame) -> None
        '''
        Constructs a Tileset instance.

        Args:
            data (DataFrame): DataFrame with width, height, depth columns.

        Returns:
            Tileset: Tileset instance.
        '''
        self.validate(data)
        data.sort_values(['width', 'height', 'depth'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        self._data = data
        self.line_color = BasicColor.CYAN
        self.line_width = 1

    def __repr__(self):
        # type: () -> str
        '''
        String representation of Tileset.
        Includes:

            * class name
            * shape
            * shape_in_tiles
            * tile_shape
            * coordinates
        '''
        return f'''
<Tileset>
         shape: {self.shape}
shape_in_tiles: {self.shape_in_tiles}
    tile_shape: {self.tile_shape}
   coordinates: {self.coordinates}'''[1:]

    @staticmethod
    def validate(data):
        # type: (DataFrame) -> None
        '''
        Validates given tileset data.

        Args:
            data (DataFrame): Tileset DataFrame.

        Raises:
            AttributeError: If data is missing width, height, depth, tile_width,
                tile_height, num_channels, bit_depth columns.
            EnforceError: If width, height, depth coordinate are not dense
                integer triplets greater than or equal to (0, 0, 0).
            EnforceError: If tile_height, tile_width, num_channels and bit_depth
                are not homogenous.
            EnforceError: If bit depth is illegal.
            EnforceError: If content column exists and is filled by anything
                other than Image instances.
        '''
        proto = utils.to_prototype(data)

        # check column names
        cols = [
            'width', 'height', 'depth', 'tile_width', 'tile_height',
            'num_channels', 'bit_depth'
        ]
        diff = set(cols).difference(list(proto.keys()))  # type: Any
        diff = sorted(list(diff))
        if len(diff) > 0:
            msg = f'Data is missing columns: {diff}.'
            raise EnforceError(msg)

        w = proto['width']
        h = proto['height']
        d = proto['depth']
        tw = proto['tile_width']
        th = proto['tile_height']
        c = proto['num_channels']

        # check bit depth
        bit_depth = data.bit_depth.tolist()
        eft.enforce_homogenous_type(bit_depth, name='Bit_depth column')
        uniq = sorted(list(set(bit_depth)))
        msg = f'Multiple bit depths found: {uniq}.'
        Enforce(len(uniq), '==', 1, message=msg)
        msg = 'Illegal bit depth: {a} not in {b}.'
        Enforce(bit_depth[0], 'in', [x.name for x in BitDepth], message=msg)

        # check coordinates density
        coords = sorted(list(zip(w, h, d)))
        eft.enforce_coordinates(coords)
        eft.enforce_dense_coordinates(coords)
        eft.enforce_coordinate_minimum(coords, (0, 0, 0))

        # check tile shape
        shapes = set(list(zip(tw, th, c)))  # type: Any
        if len(shapes) != 1:
            shapes = sorted(list(shapes))
            msg = 'All tiles must only have one shape. '
            msg += f'Found shapes: {shapes}.'
            raise EnforceError(msg)

        # validate image content
        if 'content' in proto.keys():
            imgs = proto['content']
            eft.enforce_non_empty(imgs, 'Content column')
            eft.enforce_homogenous_type(imgs, name='Content column')
            msg = 'Content column contains objects that are not instances of '
            msg += f'Image. {type(imgs[0])} != {Image}.'
            Enforce(imgs[0], 'instance of', Image, message=msg)

    # PROPERTIES----------------------------------------------------------------
    @property
    def data(self):
        # type: () -> DataFrame
        '''
        DataFrame: Copy of Tileset DataFrame.
        '''
        data = self._data
        if 'content' in data.columns:
            data = data.drop('content', axis=1).copy()
            data['content'] = self._data.content
            return data
        return self._data.copy()

    def __getitem__(self, index):
        # type: (Union[tuple, list, slice, int]) -> Tileset
        '''
        Get subset of tileset.

        Args:
            index (tuple, list, slice or int): Iterable, slice or integer of
                index values.

        Raises:
            IndexError: If index is over three dimensions.
            KeyError: If not tiles were found for given index.

        Returns:
            Tileset: Subset of Tileset.
        '''
        ind = index
        if isinstance(ind, slice) or isinstance(ind, int):
            ind = [ind]
        else:
            ind = list(ind)

        # index repr formatting
        index_repr = []  # type: Any
        for i in ind:
            if isinstance(i, slice):
                start = i.start or ''
                stop = i.stop or ''
                i = f'{start}:{stop}'
            index_repr.append(str(i))
        index_repr = '[' + ', '.join(index_repr) + ']'

        if len(ind) > 3:
            msg = 'Index must be no more than three dimensional. '
            msg += f'Given index: {index_repr}.'
            raise IndexError(msg)

        dims = ['width', 'height', 'depth']
        for i in range(3 - len(ind)):
            ind.append(None)
        lut = dict(zip(dims, ind))

        data = self.data
        for dim in dims:
            val = lut[dim]
            if val is None:
                continue
            if isinstance(val, int):
                data = data[data[dim] == val]
            elif val.start is not None:
                data = data[data[dim] >= val.start]
            elif val.stop is not None:
                data = data[data[dim] < val.stop]

        if len(data) == 0:
            msg = f'No tiles found. Given index: {index_repr}.'
            raise KeyError(msg)

        return Tileset(data)

    def get_tile_data(self, coordinate):
        # type: (Tuple[int, int, int]) -> DataFrame
        '''
        Retrieves tile data given a (width, height, depth) coordinate.

        Args:
            coordinate (tuple[int]): (width, height, depth) tuple.

        Raises:
            EnforceError: If coordinate is not a (int, int, int) tuple.
            KeyError: If no tile could be found for given coordinate.
            KeyError: If multiple tiles found for given coordinate.

        Returns:
            DataFrame: Subset of DataFrame that matches given tile coordinate.
        '''
        msg = f'Coordinate is not a tuple. Given value: {coordinate}.'
        Enforce(coordinate, 'instance of', tuple, message=msg)
        msg = f'Coordinate must be 3 dimensional. Given value: {coordinate}.'
        Enforce(len(coordinate), '==', 3, message=msg)
        msg = 'All coordinate values must be integers. Found: {a}.'
        for i in coordinate:
            Enforce(i.__class__.__name__, '==', 'int', message=msg)
        # ----------------------------------------------------------------------

        w, h, d = coordinate
        data = self.data
        data = data[data.width == w]
        data = data[data.height == h]
        data = data[data.depth == d]
        if len(data) == 0:
            msg = f'No tile found for coordinate: {coordinate}.'
            raise KeyError(msg)

        if len(data) > 1:
            msg = f'Multiple tiles found for coordinate: {coordinate}.'
            raise KeyError(msg)

        return data

    def get_tile(self, coordinate):
        # type: (Tuple[int, int, int]) -> Image
        '''
        Retrieve a tile given a (width, height, depth) coordinate.
        Loads tile content if not loaded.

        Args:
            coordinate (tuple[int]): (width, height, depth) tuple.

        Returns:
            Image: Tile of coordinate.
        '''
        data = self.get_tile_data(coordinate)
        if 'content' not in data.columns:
            data = plt.load_content(data)
        return data.content.tolist()[0]

    def get_frame_data(self, depth):
        # type: (int) -> DataFrame
        '''
        Gets subset of data that matches given depth.

        Args:
            depth (int): Depth.

        Raises:
            IndexError: If invalid depth given.

        Returns:
            DataFrame: DataFrame of matching depth.
        '''
        # validate depth
        if depth not in self.frames:
            msg = f'{depth} is not a valid depth.'
            raise IndexError(msg)

        # mask data at depth
        data = self.data
        return data[data.depth == depth]

    def get_frame(self, depth, grid=False):
        # type: (int, bool) -> Image | None
        '''
        Stitch tiles of a given depth into a single image.
        Loads tile content for frame if not loaded.
        Overlays grid onto resultant image if grid is set to True.

        Args:
            depth (int): Depth.
            grid (bool, optional): If true, overlay grid on resulting image.
                Default: False.

        Returns:
            Image: Image of stitched tiles at given depth.
        '''
        data = self.get_frame_data(depth)
        if 'content' not in data.columns:
            data = plt.load_content(data)

        x0 = int(data.width.min())
        x1 = int(data.width.max())
        y0 = int(data.height.min())
        y1 = int(data.height.max())

        # build columns first then staple them horizontally
        image = None
        for x in range(x0, x1 + 1):
            col = self.get_tile((x, y0, depth))
            for y in range(y0 + 1, y1 + 1):
                col = ops.edit.staple(col, self.get_tile((x, y, depth)), 'above')
            if image is None:
                image = col
            else:
                image = ops.edit.staple(image, col, direction='right')

        # overlay grid
        if grid and image is not None:
            image = ops.draw.grid(
                image,
                (self.width_in_tiles, self.height_in_tiles),
                self.line_color,
                self.line_width,
            )
        return image

    @property
    def tile_shape(self):
        # type: () -> Tuple[int, int, int]
        '''
        tuple[int]: (tile width, tile height, num_channels).
        '''
        x = self._data.loc[0]
        return int(x.tile_width), int(x.tile_height), int(x.num_channels)

    @property
    def tile_width(self):
        # type: () -> int
        '''
        int: Tile width in pixels.
        '''
        return int(self._data.loc[0, 'tile_width'])

    @property
    def tile_height(self):
        # type: () -> int
        '''
        int: Tile height in pixels.
        '''
        return int(self._data.loc[0, 'tile_height'])

    @property
    def width(self):
        # type: () -> int
        '''
        int: Whole tileset width in pixels.
        '''
        return self.width_in_tiles * self.tile_width

    @property
    def width_in_tiles(self):
        # type: () -> int
        '''
        int: Whole tileset width in tiles.
        '''
        return (self._data.width.max() + 1) - self._data.width.min()

    @property
    def height(self):
        # type: () -> int
        '''
        int: Whole tileset height in pixels.
        '''
        return self.height_in_tiles * self.tile_height

    @property
    def height_in_tiles(self):
        # type: () -> int
        '''
        int: Whole tileset height in tiles.
        '''
        return (self._data.height.max() + 1) - self._data.height.min()

    @property
    def depth(self):
        # type: () -> int
        '''
        int: Whole tileset depth in tiles.
        '''
        return (self._data.depth.max() + 1) - self._data.depth.min()

    @property
    def frames(self):
        # type: () -> List[int]
        '''
        list[int]: List of tileset depths.
        '''
        return sorted(self._data.depth.unique().tolist())

    @property
    def num_channels(self):
        # type: () -> int
        '''
        int: Number of channels for all tiles.
        '''
        return int(self._data.loc[0, 'num_channels'])

    @property
    def shape(self):
        # type: () -> Tuple[int, int, int, int]
        '''
        tuple[int]: (width, height, depth, num_channels).
        '''
        return (self.width, self.height, self.depth, self.num_channels)

    @property
    def shape_in_tiles(self):
        # type: () -> Tuple[int, int, int]
        '''
        tuple[int]: (width, height, depth) in tiles.
        '''
        return (self.width_in_tiles, self.height_in_tiles, self.depth)

    @property
    def coordinates(self):
        # type: () -> List[Tuple[int, int, int]]
        '''
        Returns a list of all tile coordinates.

        Returns:
            list[tuple]: List of (width, height, depth) tuples.
        '''
        return self._data \
            .apply(lambda x: (x.width, x.height, x.depth), axis=1) \
            .tolist()

    def apply(
        self,
        function  # type: Callable[[Tuple[int, int], Image], Tuple[Tuple[int, int], Image]]
    ):  # type: (...) -> Tileset
        '''
        Applies a given function to all the tiles within the tileset.

        Args:
            function (function): Function of form:
                lambda (int, int), Image: (int, int), Image.

        Returns:
            Tileset: New Tileset with function applied to tiles.
        '''
        data = utils.xstarmap(function, self.to_dict().items())
        return Tileset.from_dict(dict(data))

    # IMPORT-EXPORT-------------------------------------------------------------
    @staticmethod
    def from_image(image, shape, anchor='bottom-left', color=BasicColor.BLACK):
        # type: (Image, Tuple[int, int], str, Union[Color, BasicColor]) -> Tileset
        '''
        Splits a given image into tiles of given shape.

        Anchor options include:

            * top-left
            * top-center
            * top-right
            * center-left
            * center-center
            * center-right
            * bottom-left
            * bottom-center
            * bottom-right

        Args:
            image (Image): Image to be tiled.
            shape (tuple[int]): Tile width and height.
            anchor (str, optional): How the given image will first be padded.
                Default: bottom-left.
            color (Color or BasicColor, optional): Color of image padding.

        Raises:
            EnforceError: If image is not an instance of Image.
            EnforceError: If shape is not a tuple of 2 or more integers greater
                than 0.

        Returns:
            Tileset: Tileset instance.
        '''
        # enforcements
        Enforce(image, 'instance of', Image)
        eft.enforce_2d_shape(shape)
        # ----------------------------------------------------------------------

        tw = shape[0]
        th = shape[1]

        w, h, c = image.shape
        w = math.ceil(w / tw) * tw
        h = math.ceil(h / th) * th
        image = ops.edit.pad(image, (w, h, c), anchor=anchor, color=color)

        w, h, c = image.shape
        cols = []
        for x in range(tw, w, tw):
            col, image = ops.edit.cut(image, tw, axis='vertical')
            cols.append(col)
        cols.append(image)

        data_ = []
        max_y = len(list(range(0, h, th))) - 1
        for x, col in enumerate(cols):
            for y, _ in enumerate(range(th, h, th)):
                tile, col = ops.edit.cut(col, th, axis='horizontal')
                data_.append([x, max_y - y, tile])
            data_.append([x, 0, col])
        data = DataFrame(data_, columns=['width', 'height', 'content'])
        data['depth'] = 0
        data['tile_width'] = tw
        data['tile_height'] = th
        data['num_channels'] = c
        data['bit_depth'] = image.bit_depth.name

        return Tileset(data)

    @staticmethod
    def from_images(
        images,                 # type: List[Image]
        shape,                  # type: Tuple[int, int]
        anchor='bottom-left',   # type: str
        color=BasicColor.BLACK  # type: Union[Color, BasicColor]
    ):  # type: (...) -> Tileset
        '''
        Splits each image of a given list into tiles of given shape.

        Anchor options include:

            * top-left
            * top-center
            * top-right
            * center-left
            * center-center
            * center-right
            * bottom-left
            * bottom-center
            * bottom-right

        Args:
            images (list[Image]): Images to be tiled.
            shape (tuple[int]): Tile width and height.
            anchor (str, optional): How the given image will first be padded.
                Default: bottom-left.
            color (Color or BasicColor, optional): Color of image padding.

        Raises:
            EnforceError: If images is not a list of Image instances.
            EnforceError: If shape is not a tuple of 2 or more integers greater
                than 0.

        Returns:
            Tileset: Tileset instance.
        '''
        # enforcements
        msg = 'Images must be a list of Image instances. {a} != {b}.'
        Enforce(images, 'instance of', list, message=msg)
        for img in images:
            Enforce(img, 'instance of', Image, message=msg)

        eft.enforce_2d_shape(shape)
        # ----------------------------------------------------------------------

        data = []
        for i, img in enumerate(images):
            datum = Tileset.from_image(img, shape, anchor=anchor, color=color)._data
            datum['depth'] = i
            data.append(datum)
        data = pd.concat(data)
        return Tileset(data)

    def to_images(self):
        # type: () -> List[Image | None]
        '''
        Combines tiles into list of images.

        Returns:
            list[Image]: List of stitched tile images, one per frame.
        '''
        return [self.get_frame(f) for f in self.frames]

    @staticmethod
    def from_dict(data):
        # type: (Dict[Tuple[int, int, int], Image]) -> Tileset
        '''
        Construct a Tileset from a given dictionary.
        Dictionary is of the form: {(int, int, int): Image}.
        Key order is (width, height, depth).

        Args:
            data (dict): Tile dictionary.

        Raises:
            EnforceError: If data is not a {(int, int, int): Image} dictionary.

        Returns:
            Tileset: Tileset of dictionary.
        '''
        Enforce(data, 'instance of', dict)
        for key in data.keys():
            msg = f'Key: {key} is not a tuple of three integers.'
            Enforce(key, 'instance of', tuple, message=msg)
            Enforce(len(key), '==', 3, message=msg)
            Enforce(key[0], 'instance of', int, message=msg)
            Enforce(key[1], 'instance of', int, message=msg)
            Enforce(key[2], 'instance of', int, message=msg)
        # ----------------------------------------------------------------------

        temp = []
        for (w, h, d), img in data.items():
            datum = dict(
                width=w,
                height=h,
                depth=d,
                content=img,
                tile_width=img.width,
                tile_height=img.height,
                num_channels=img.num_channels,
                bit_depth=img.bit_depth.name,
            )
            temp.append(datum)
        data = DataFrame(temp)
        return Tileset(data)

    def to_dict(self):
        # type: () -> Dict[Tuple[int, int, int], Image]
        '''
        Returns a (width, height, depth): Image dictionary of tiles.

        Returns:
            dict: Dictionary representation of tileset.
        '''
        data = self._data
        if 'content' not in self._data.columns:
            data = plt.load_content(data)

        output = data \
            .apply(lambda x: ((x.width, x.height, x.depth), x.content), axis=1) \
            .tolist()
        output = dict(output)
        return output

    @staticmethod
    def read_meta(directory, specification=None, trait_sampling=False):
        # type: (Union[str, Path], Optional[SpecificationBase], bool) -> Tileset
        '''
        Generates data for a given directory of tiled image files.
        Filenames must be of the form w-h-d. Where w is width, h is height, and
        d is depth. W, h and d must be quadruple padded numbers separated by
        hyphens. Data will be validated according to specification.
        Filename examples:

            * foobar_0000-0001-0002.png
            * foobar_0000-0001-0002_foo.png
            * foobar_c0000-0001-0002.png
            * 0000-0001-0002.png
            * 0000-0001-0002_foo.png

        Args:
            directory (str or Path): Directory of tiles to be read.
            specification (SpecificationBase, optional): Asset specification
                class. Default: Tileset.SPECIFICATION.
            trait_sampling (bool, optional): If true will sample file traits
                and assume the traits returned are the same for all files.
                Default: False.

        Raises:
            EnforceError: If spec is not a subclass of SpecificationBase.
            EnforceError: If specification does not define the following
                attributes:

                * tile_width
                * tile_height
                * num_channels
                * bit_depth

            FileNotFoundError: If not tile files are found in given directory.

        Returns:
            Tileset: Tileset without content loaded.
        '''
        spec = Tileset.SPECIFICATION if specification is None else specification
        msg = f'{spec} is not a subclass of SpecificationBase.'
        Enforce(
            issubclass(spec, SpecificationBase), '==', True,
            message=msg
        )

        msg = '{} must be a class member included in your specification.'
        attrs = [
            'tile_width',
            'tile_height',
            'num_channels',
            'bit_depth',
        ]
        for attr in attrs:
            Enforce(
                hasattr(spec, attr), '==', True,
                message=msg.format(attr.capitalize()),
            )
        # ----------------------------------------------------------------------

        data = hbt.directory_to_dataframe(directory)
        regex = r'(\d\d\d\d)-(\d\d\d\d)-(\d\d\d\d)'
        mask = data.filename.apply(lambda x: re.search(regex, x)).astype(bool)
        data = data[mask]
        data.reset_index(drop=True, inplace=True)
        data['uri'] = data.filepath.apply(lambda x: 'file://' + x)
        data['content_type'] = 'image'
        del data['extension']

        if len(data) == 0:
            msg = f'No tile files found in directory: {directory}.'
            raise FileNotFoundError(msg)

        # width
        data['width'] = data.filepath \
            .apply(lambda x: re.search(regex, x).group(1))  # type: ignore
        data.width = data.width.apply(int)

        # height
        data['height'] = data.filepath \
            .apply(lambda x: re.search(regex, x).group(2))  # type: ignore
        data.height = data.height.apply(int)

        # depth
        data['depth'] = data.filepath \
            .apply(lambda x: re.search(regex, x).group(3))  # type: ignore
        data.depth = data.depth.apply(int)

        # trait sampling is needed for assets with many files and/or very large
        # files
        # unfortunately, trait sampling cannot be tested accurately because the
        # distribution of invalid files per asset is unknown
        if trait_sampling:
            data = utils \
                .sample_traits(data, spec, sample_size=Tileset.SAMPLE_SIZE)  # pragma: no cover
        else:
            data = utils.add_traits(data, spec)

        utils.validate_with_specification(data, spec)

        # organize columns
        cols = [
            'width',
            'height',
            'depth',
            'tile_height',
            'tile_width',
            'num_channels',
            'bit_depth',
        ]
        cols += spec.filename_fields
        cols += ['uri', 'content_type']
        data = data[cols]
        return Tileset(data)

    @staticmethod
    def read(directory, specification=None):
        # type: (Union[str, Path], Optional[SpecificationBase]) -> Tileset
        '''
        Generates data and loads content for a given directory of tiled image
        files. Filenames must be of the form w-h-d. Where w is width, h is
        height, and d is depth. W, h and d must be quadruple padded numbers
        separated by hyphens. Data will be validated according to specification.
        Filename examples:

            * foobar_0000-0001-0002.png
            * foobar_0000-0001-0002_foo.png
            * foobar_c0000-0001-0002.png
            * 0000-0001-0002.png
            * 0000-0001-0002_foo.png

        Args:
            directory (str or Path): Directory of tiles to be read.
            specification (SpecificationBase, optional): Asset specification
                class. Default: Tileset.SPECIFICATION.

        Raises:
            FileNotFoundError: If not tile files are found in given directory.
        '''
        data = Tileset.read_meta(directory, specification=specification)._data
        data = plt.load_content(data)
        return Tileset(data)

    def write(
        self,
        target,                # type: Union[str, Path]
        specification=None,    # type: Optional[SpecificationBase]
        codec=ImageCodec.PIZ,  # type: ImageCodec
    ):                         # type: (...) -> Tileset
        '''
        Writes tiles to a given target directory.

        Args:
            target (str or Path): Directory in which file will be written.
            specification (SpecificationBase, optional): Asset specification
                class to be used in generating filepaths.
                Default: Tileset.SPECIFICATION.
            codec (ImageCodec): EXR codec to be used. Default: PIZ.

        Returns:
            Tileset: Self.
        '''
        spec = specification
        if spec is None:
            spec = Tileset.SPECIFICATION
        utils.validate_with_specification(self._data, spec)

        data = self.data
        os.makedirs(target, exist_ok=True)
        proto = utils.to_prototype(data)
        keys = set(proto.keys()).intersection(spec.fields.keys())
        proto = {k: proto[k] for k in keys}

        data['target'] = spec(proto).to_filepaths(target)
        data.target.apply(lambda x: os.makedirs(Path(x).parent, exist_ok=True))
        data.apply(
            lambda x: x.content.write(x.target, codec=codec),
            axis=1
        )
        return self
