from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hidebound.core.specification_base import ComplexSpecificationBase, SpecificationBase
from lunchbox.enforce import EnforceError
from pandas import DataFrame
from schematics.exceptions import DataError
from schematics.types import IntType, ListType, StringType
import hidebound.core.validators as vd
import lunchbox.tools as lbt
import pandas as pd
import pytest

from vision.enforce.enforce_image_attributes import EnforceImageAttributes
from vision.enforce.enforce_image_content import EnforceImageContent
from vision.image.color import BasicColor
from vision.image.image import BitDepth, Image
from vision.image.sequence import ImageSequence
from vision.image.tileset import Tileset
import vision.image.image_tools as imt
import vision.pipeline.traits as tr
# ------------------------------------------------------------------------------


class FakeTileSetSpec(ComplexSpecificationBase):
    asset_name_fields = ['project', 'specification', 'descriptor', 'version']
    filename_fields = [
        'project', 'specification', 'descriptor', 'version', 'coordinate',
        'extension'
    ]
    tile_width = ListType(
        IntType(), required=True, validators=[lambda x: vd.is_eq(x, 200)]
    )
    tile_height = ListType(
        IntType(), required=True, validators=[lambda x: vd.is_eq(x, 200)]
    )
    num_channels = ListType(
        IntType(), required=True, validators=[lambda x: vd.is_eq(x, 3)]
    )
    bit_depth = ListType(
        StringType(), required=True, validators=[lambda x: vd.is_eq(x, 'UINT8')]
    )
    coordinate = ListType(
        ListType(
            IntType(),
            validators=[
                vd.is_coordinate,
                lambda x: vd.is_eq(len(x), 3),
            ]
        ),
        validators=[
            vd.has_uniform_coordinate_count,
            vd.has_dense_coordinates,
            lambda x: vd.coordinates_begin_at(x, [0, 0, 0]),
        ],
        required=True,
    )
    extension = ListType(
        StringType(),
        required=True,
        validators=[vd.is_extension, lambda x: vd.is_eq(x, 'png')]
    )
    file_traits = dict(
        tile_width=tr.get_image_width,
        tile_height=tr.get_image_height,
        num_channels=tr.get_num_image_channels,
        bit_depth=tr.get_image_bit_depth,
    )

    def to_filepaths(self, root):
        asset, filename = self.get_name_patterns()
        pattern = Path(asset, 'f{coordinate[2]:04d}', filename).as_posix()
        return self._to_filepaths(root, pattern)


class TilesetTests(unittest.TestCase):
    def get_uv_checker_image(self):
        tgt = lbt.relative_path(__file__, '../../../resources/uv-checker.png')
        return Image.read(tgt)

    def get_uv_checker_tileset(self):
        img = self.get_uv_checker_image()
        output = Tileset.from_image(img, (200, 200))
        output._data['project'] = 'proj001'
        output._data['specification'] = 'tset001'
        output._data['descriptor'] = 'desc'
        output._data['version'] = 1
        output._data['coordinate'] = output._data \
            .apply(lambda x: [x.width, x.height, x.depth], axis=1)
        output._data['extension'] = 'png'
        return output

    def get_image(self):
        # 150 x 200 image with 64 x 64 tiles --> 192 x 256 image with 3 x 4 tiles
        img = imt.get_swatch((150, 200, 3), BasicColor.BLACK)
        img = imt.pad(img, (150, 205, 3), anchor='top-left', color=BasicColor.RED)
        img = imt.pad(img, (150, 210, 3), anchor='bottom-left', color=BasicColor.WHITE)
        img = imt.pad(img, (155, 210, 3), anchor='top-left', color=BasicColor.BLUE)
        img = imt.pad(img, (160, 210, 3), anchor='top-right', color=BasicColor.GREEN)
        return img

    def get_images(self):
        return [self.get_image(), self.get_image()]

    def get_padded_image(self):
        return imt.pad(self.get_image(), (192, 256, 3), anchor='bottom-left')

    def get_data(self):
        image = self.get_padded_image()

        # create columns
        tw = 64
        th = 64
        w = 192
        h = 256
        cols = []
        for x in range(tw, w, tw):
            col, image = imt.cut(image, tw, axis='vertical')
            cols.append(col)
        cols.append(image)

        # create tiles and add them to data
        data = []
        max_y = len(list(range(0, h, th))) - 1
        for x, col in enumerate(cols):
            for y, row in enumerate(range(th, h, th)):
                tile, col = imt.cut(col, th, axis='horizontal')
                data.append([x, max_y - y, tile])
            data.append([x, 0, col])
        data = DataFrame(data, columns=['width', 'height', 'content'])

        # duplicate data along depth axis
        data['depth'] = 0
        data['bit_depth'] = image.bit_depth.name
        data['tile_height'] = th
        data['tile_width'] = tw
        data['num_channels'] = image.num_channels
        b = data.copy()
        b['depth'] = 1
        data = pd.concat([data, b], axis=0, ignore_index=True)

        data.sort_values(['width', 'height', 'depth'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # ensure coordinates are correct
        result = data.apply(lambda x: (x.width, x.height, x.depth), axis=1)
        result = sorted(result.tolist())
        expected = sorted(list(product(range(3), range(4), range(2))))
        self.assertEqual(result, expected)

        return data

    def write_data_content(self, data, root):
        def get_filepath(row, root):
            filename = f'{row.height:04d}-{row.width:04d}-{row.depth:04d}.exr'
            filepath = Path(root, filename).as_posix()
            return filepath

        data['content_type'] = 'image'
        data['filepath'] = data.apply(lambda x: get_filepath(x, root), axis=1)
        data['uri'] = data.filepath.apply(lambda x: 'file://' + x)
        data.apply(lambda x: x.content.write(x.filepath), axis=1)

        del data['filepath']
        return data

    def write_tileset_data(self, data, root):
        name = 'p-proj001_s-tset001_d-desc_v001'
        data.depth.apply(
            lambda x: os.makedirs(Path(root, name, f'f{x:04d}'), exist_ok=True)
        )
        data.apply(
            lambda x: x.content.write(Path(
                root, name, f'f{x.depth:04d}',
                f'{name}_c{x.width:04d}-{x.height:04d}-{x.depth:04d}.png'
            )),
            axis=1
        )

    # IMPORT-EXPORT-------------------------------------------------------------
    def test_init(self):
        expected = self.get_data()
        result = Tileset(expected)
        self.assertEqual(
            result._data.columns.tolist(),
            expected.columns.tolist()
        )
        self.assertEqual(result._data.shape, expected.shape)

    def test_repr(self):
        expected = r'''<Tileset>
         shape: \(192, 256, 2, 3\)
shape_in_tiles: \(3, 4, 2\)
    tile_shape: \(64, 64, 3\)
   coordinates: \[\(0, 0, 0\).*\(2, 3, 1\)\]'''

        result = Tileset(self.get_data())
        self.assertRegex(result.__repr__(), expected)

    def test_from_image(self):
        img = self.get_image()
        result = Tileset.from_image(img, (64, 64, 3), anchor='bottom-left')

        expected = self.get_data()
        expected = expected[expected.depth == 0]
        expected.reset_index(drop=True, inplace=True)
        for i, row in expected.iterrows():
            tile = result._data.loc[i, 'content']
            self.assertEqual(tile.shape, (64, 64, 3))
            EnforceImageContent(tile, '==', row.content)

    def test_from_image_errors(self):
        expected = 'None is not instance of .*Image'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_image(None, (64, 64, 3))

        img = self.get_image()
        with self.assertRaises(EnforceError):
            Tileset.from_image(img, (64))

        with self.assertRaises(EnforceError):
            Tileset.from_image(img, ('ten', 64))

        with self.assertRaises(EnforceError):
            Tileset.from_image(img, (-5, 64))

    def test_from_images(self):
        imgs = self.get_images()
        result = Tileset.from_images(imgs, (64, 64, 3), anchor='bottom-left')

        expected = self.get_data()
        for i, row in expected.iterrows():
            tile = result._data.loc[i, 'content']
            self.assertEqual(tile.shape, (64, 64, 3))
            EnforceImageContent(tile, '==', row.content)

    def test_from_images_errors(self):
        expected = 'Images must be a list of Image instances. None != .*list'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_images(None, (64, 64, 3))

        expected = 'Images must be a list of Image instances. None != .*Image'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_images([None, None], (64, 64, 3))

        imgs = self.get_images()
        with self.assertRaises(EnforceError):
            Tileset.from_images(imgs, (64))

        with self.assertRaises(EnforceError):
            Tileset.from_images(imgs, ('ten', 64))

        with self.assertRaises(EnforceError):
            Tileset.from_images(imgs, (-5, 64))

    def test_from_image_sequence(self):
        imgs = ImageSequence.from_images(self.get_images())
        result = Tileset.from_image_sequence(imgs, (64, 64, 3), anchor='bottom-left')

        expected = self.get_data()
        for i, row in expected.iterrows():
            tile = result._data.loc[i, 'content']
            self.assertEqual(tile.shape, (64, 64, 3))
            EnforceImageContent(tile, '==', row.content)

    def test_from_image_sequence_errors(self):
        expected = 'None is not instance of .*ImageSequence'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_image_sequence(None, (64, 64, 3))

        imgs = ImageSequence.from_images(self.get_images())
        with self.assertRaises(EnforceError):
            Tileset.from_image_sequence(imgs, (64))

        with self.assertRaises(EnforceError):
            Tileset.from_image_sequence(imgs, ('ten', 64))

        with self.assertRaises(EnforceError):
            Tileset.from_image_sequence(imgs, (-5, 64))

    def test_to_images(self):
        # imgs are (160, 210, 3)
        imgs = self.get_images()
        shape = (64, 64, 3)
        results = Tileset.from_images(imgs, shape).to_images()
        self.assertIsInstance(results, list)

        expected = self.get_padded_image()
        for result in results:
            EnforceImageAttributes(result, '==', expected, 'shape')
            EnforceImageContent(result, '==', expected)

    def test_to_image_sequence(self):
        # imgs are (160, 210, 3)
        imgs = self.get_images()
        shape = (64, 64, 3)
        results = Tileset.from_images(imgs, shape).to_image_sequence()
        self.assertIsInstance(results, ImageSequence)

        expected = self.get_padded_image()
        for result in results:
            EnforceImageAttributes(result, '==', expected, 'shape')
            EnforceImageContent(result, '==', expected)

    def test_from_dict(self):
        data = self.get_data()
        keys = data \
            .apply(lambda x: (x.width, x.height, x.depth), axis=1) \
            .tolist()
        vals = data.content.tolist()
        data = dict(zip(keys, vals))

        result = Tileset.from_dict(data)
        for key, val in data.items():
            self.assertIs(result.get_tile(key), val)

    def test_from_dict_errors(self):
        with self.assertRaises(EnforceError):
            Tileset.from_dict('foo')

        expected = 'Key: foo is not a tuple of three integers.'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_dict({'foo': 'bar'})

        expected = r'Key: \(0, 1\) is not a tuple of three integers\.'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_dict({(0, 1): 'bar'})

        expected = r"Key: \('foo', 1, 0\) is not a tuple of three integers\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_dict({('foo', 1, 0): 'bar'})

        expected = r"Key: \(1, 2, 'foo'\) is not a tuple of three integers\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.from_dict({(1, 2, 'foo'): 'bar'})

    def test_to_dict(self):
        data = self.get_data()
        keys = data.apply(lambda x: (x.width, x.height, x.depth), axis=1).tolist()
        vals = data.content.tolist()
        data = dict(zip(keys, vals))

        result = Tileset.from_dict(data).to_dict()
        for key, val in data.items():
            self.assertIs(result[key], val)

        with TemporaryDirectory() as root:
            data = self.get_data()
            data = self.write_data_content(data, root)
            del data['content']
            result = Tileset(data).to_dict()

            keys = data.apply(lambda x: (x.width, x.height, x.depth), axis=1).tolist()
            vals = data.content.tolist()
            data = dict(zip(keys, vals))

            for key, val in data.items():
                EnforceImageContent(result[key], '==', val)

    # READ-WRITE----------------------------------------------------------------
    def test_read(self):
        data = self.get_uv_checker_tileset()._data
        images = self.get_uv_checker_tileset().to_dict().items()
        with TemporaryDirectory() as root:
            self.write_tileset_data(data, root)

            result = Tileset.read(root, FakeTileSetSpec)
            for (x, y, z), expected in images:
                EnforceImageContent(result.get_tile((x, y, z)), '==', expected)

    def test_read_meta(self):
        data = self.get_uv_checker_tileset()._data
        with TemporaryDirectory() as root:
            self.write_tileset_data(data, root)

            results = Tileset.read_meta(root, FakeTileSetSpec)._data
            for col in data.drop('content', axis=1).columns:
                result = results[col].tolist()
                expected = data[col].tolist()
                self.assertEqual(result, expected)

    def test_read_meta_errors(self):
        img = imt \
            .get_swatch((5, 5, 3), BasicColor.BLACK) \
            .to_bit_depth(BitDepth.UINT8)
        with TemporaryDirectory() as root:
            expected = f'No tile files found in directory: {root}.'
            with self.assertRaisesRegexp(FileNotFoundError, expected):
                Tileset.read_meta(root)

            img.write(Path(root, 'badname_01-02-03.png'))
            with self.assertRaisesRegexp(FileNotFoundError, expected):
                Tileset.read_meta(root)

            img.write(Path(root, 'badname_001-002.png'))
            with self.assertRaisesRegexp(FileNotFoundError, expected):
                Tileset.read_meta(root)

    def test_read_meta_spec_errors(self):
        class BadSpec():
            pass
        expected = '.* is not a subclass of SpecificationBase.'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.read_meta('/dir', specification=BadSpec)

        class BadSpec(SpecificationBase):
            tile_width = ListType(IntType())
            tile_height = ListType(IntType())
            num_channels = ListType(IntType())
            extension = ListType(StringType())
            file_traits = dict(
                tile_width=tr.get_image_width,
                tile_height=tr.get_image_height,
                num_channels=tr.get_num_image_channels,
            )
        expected = 'Bit_depth must be a class member included in your '
        expected += 'specification.'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.read_meta('/dir', specification=BadSpec)

    def test_read_meta_file_traits_error(self):
        data = self.get_uv_checker_tileset()._data
        for i, row in data.iterrows():
            data.loc[i, 'content'] = imt \
                .get_swatch((i + 1, 10, 3), BasicColor.BLACK) \
                .to_bit_depth(BitDepth.UINT8)

        with TemporaryDirectory() as root:
            self.write_tileset_data(data, root)
            expected = r'\{"tile_width": \["1 != 1024."\], '
            expected += r'"tile_height": \["10 != 1024."\]\}'
            with self.assertRaisesRegexp(DataError, expected):
                Tileset.read_meta(root)

    @pytest.mark.skipif('SKIP_SLOW_TESTS' in os.environ, reason='slow test')
    def test_write(self):
        expected = self.get_uv_checker_tileset()
        with TemporaryDirectory() as root:
            spec_path = Path(root, 'tile001')
            expected.write(spec_path, FakeTileSetSpec)

            self.assertTrue(os.path.exists(spec_path))
            results = Tileset.read(spec_path, FakeTileSetSpec).to_dict().items()
            for coord, result in results:
                EnforceImageContent(result, '==', expected.get_tile(coord))

    @pytest.mark.skipif('SKIP_SLOW_TESTS' in os.environ, reason='slow test')
    def test_write_no_spec(self):
        data = self.get_uv_checker_tileset().data
        w, h, _ = data.content[0].shape
        w = 1024 / w
        h = 1024 / h
        data.content = data.content.apply(lambda x: imt.reformat(x, w, h))
        data.tile_width = data.content.apply(lambda x: x.width)
        data.tile_height = data.content.apply(lambda x: x.height)
        expected = Tileset(data)

        with TemporaryDirectory() as root:
            spec_path = Path(root, 'tile001')
            expected.write(spec_path)

            self.assertTrue(os.path.exists(spec_path))
            results = Tileset.read(spec_path).to_dict().items()
            for coord, result in results:
                EnforceImageContent(result, '==', expected.get_tile(coord))

    # VALIDATE------------------------------------------------------------------
    def test_validate(self):
        data = self.get_data()
        Tileset.validate(data)

        del data['content']
        Tileset.validate(data)

    def test_validate_missing_columns(self):
        cols = [
            'width', 'height', 'depth', 'tile_width', 'tile_height',
            'num_channels', 'bit_depth'
        ]
        data = self.get_data()
        for col in cols:
            d = data.copy()
            del d[col]
            expected = f"Data is missing columns: \['{col}'\]\."  # noqa: W605
            with self.assertRaisesRegexp(EnforceError, expected):
                Tileset.validate(d)

    def test_validate_bit_depth(self):
        data = self.get_data()
        data.loc[0, 'bit_depth'] = 999
        expected = 'Bit_depth column may only contain one type of object. '
        expected += r"Found types: \['int', 'str'\]\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

        data.loc[0, 'bit_depth'] = BitDepth.FLOAT16.name
        expected = r"Multiple bit depths found: \['FLOAT16', 'FLOAT32'\]\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

        data.bit_depth = 'FOOBAR'
        expected = r"Illegal bit depth: FOOBAR not in \[.*\]\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

    def test_validate_coordinates_missing(self):
        data = self.get_data().loc[1:]
        expected = r'Missing coordinates: \[\(0, 0, 0\)\]\.'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

    def test_validate_coordinates_type(self):
        data = self.get_data()
        data.loc[0, 'height'] = 0.1
        with self.assertRaises(EnforceError):
            Tileset.validate(data)

    def test_validate_coordinate_minimum(self):
        data = self.get_data()
        data.loc[0, 'width'] = -1
        with self.assertRaises(EnforceError):
            Tileset.validate(data)

    def test_validate_tile_shape(self):
        data = self.get_data()
        data.loc[0, 'tile_height'] = 99
        expected = 'All tiles must only have one shape. Found shapes: '
        expected += r'\[\(64, 64, 3\), \(64, 99, 3\)\]'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

    def test_validate_content(self):
        data = self.get_data()
        data.loc[0, 'content'] = 'banana'
        expected = 'Content column may only contain one type of object. '
        expected += r"Found types: \['Image', 'str'\]\."
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

        data.content = 'banana'
        expected = 'Content column contains objects that are not instances of '
        expected += 'Image. .*str.* != .*Image'
        with self.assertRaisesRegexp(EnforceError, expected):
            Tileset.validate(data)

    # GET-ITEM------------------------------------------------------------------
    def test_getitem(self):
        data = self.get_data()
        tiles = Tileset(data)

        expected = sorted(list(product(range(3), [1], range(2))))
        result = tiles[:, 1]
        result = sorted(list(result.to_dict().keys()))
        self.assertEqual(result, expected)

        expected = sorted(list(product(range(3), [1], range(2))))
        result = tiles[:, 1, :]
        result = sorted(list(result.to_dict().keys()))
        self.assertEqual(result, expected)

        expected = sorted(list(product([1, 2], [0, 1], range(2))))
        result = tiles[1:, :2, :]
        result = sorted(list(result.to_dict().keys()))
        self.assertEqual(result, expected)

        expected = sorted(list(product([1], range(4), range(2))))
        result = tiles[1]
        result = sorted(list(result.to_dict().keys()))
        self.assertEqual(result, expected)

    def test_getitem_errors(self):
        tiles = self.get_uv_checker_tileset()
        expected = 'Index must be no more than three dimensional. '
        expected += r'Given index: \[:, :, :, 3\]\.'
        with self.assertRaisesRegexp(IndexError, expected):
            tiles[:, :, :, 3]

        expected = r'No tiles found. Given index: \[:, :, 999\]\.'
        with self.assertRaisesRegexp(KeyError, expected):
            tiles[:, :, 999]

    # GET-TILE------------------------------------------------------------------
    def test_get_tile(self):
        data = self.get_data()
        tileset = Tileset(data)
        e = tileset._data
        e = e[e.width == 1]
        e = e[e.height == 2]
        e = e[e.depth == 0]
        expected = e.content.tolist()[0]
        result = tileset.get_tile((1, 2, 0))
        self.assertIs(result, expected)

    def test_get_tile_no_content(self):
        data = self.get_data()
        tileset = Tileset(data)
        e = tileset._data
        e = e[e.width == 1]
        e = e[e.height == 2]
        e = e[e.depth == 0]
        expected = e.content.tolist()[0]

        with TemporaryDirectory() as root:
            data = self.write_data_content(data, root)
            del data['content']

            tgt = Path(root, 'foo.exr').as_posix()
            expected.write(tgt)
            data['uri'] = 'file://' + tgt

            result = Tileset(data).get_tile((1, 2, 0))
            EnforceImageContent(result, '==', expected)

    def test_get_tile_data(self):
        data = self.get_data()
        result = Tileset(data).get_tile_data((1, 2, 0)).index

        data = data[data.width == 1]
        data = data[data.height == 2]
        data = data[data.depth == 0]
        expected = data.index
        self.assertEqual(result, expected)

    def test_get_tile_data_errors(self):
        data = self.get_data()
        tileset = Tileset(data)

        expected = 'Coordinate is not a tuple. Given value: foo.'
        with self.assertRaisesRegexp(EnforceError, expected):
            tileset.get_tile_data('foo')

        expected = r"Coordinate must be 3 dimensional\. Given value: \(1, 2\)\."
        with self.assertRaisesRegexp(EnforceError, expected):
            tileset.get_tile_data((1, 2))

        expected = 'All coordinate values must be integers. Found: str.'
        with self.assertRaisesRegexp(EnforceError, expected):
            tileset.get_tile_data((1, 2, 'bar'))

    def test_get_tile_data_not_found(self):
        data = self.get_data()
        tileset = Tileset(data)
        expected = r'No tile found for coordinate: \(7, 10, 999\)\.'
        with self.assertRaisesRegexp(KeyError, expected):
            tileset.get_tile_data((7, 10, 999))

    def test_get_tile_data_multiple_found(self):
        data = self.get_data()
        tileset = Tileset(data)
        tileset._data.loc[1, 'width'] = 0
        tileset._data.loc[1, 'height'] = 0
        tileset._data.loc[1, 'depth'] = 0
        expected = r'Multiple tiles found for coordinate: \(0, 0, 0\)\.'
        with self.assertRaisesRegexp(KeyError, expected):
            tileset.get_tile_data((0, 0, 0))

    # GET-FRAME-----------------------------------------------------------------
    def test_get_frame_data(self):
        data = self.get_data()
        tileset = Tileset(data)

        data = tileset._data
        data = data[data.depth == 1]
        expected = data.index.tolist()

        result = tileset.get_frame_data(1).index.tolist()
        self.assertEqual(result, expected)

    def test_get_frame(self):
        data = self.get_data()
        tileset = Tileset(data)
        expected = self.get_padded_image()
        result = tileset.get_frame(1)
        EnforceImageAttributes(result, '==', expected, 'shape')
        EnforceImageContent(result, '==', expected)

    def test_get_frame_no_content(self):
        data = self.get_data()
        expected = self.get_padded_image()
        with TemporaryDirectory() as root:
            data = self.write_data_content(data, root)
            del data['content']
            result = Tileset(data).get_frame(1)
            EnforceImageAttributes(result, '==', expected, 'shape')
            EnforceImageContent(result, '==', expected)

    def test_get_frame_grid(self):
        data = self.get_data()
        tileset = Tileset(data)
        expected = self.get_padded_image()
        result = tileset.get_frame(1, grid=True)
        expected = imt.draw_grid(
            expected,
            (tileset.width_in_tiles, tileset.height_in_tiles),
            tileset.line_color,
            tileset.line_width,
        )
        EnforceImageAttributes(result, '==', expected, 'shape')
        EnforceImageContent(result, '==', expected)

    def test_get_frame_errors(self):
        data = self.get_data()
        tileset = Tileset(data)
        expected = '99 is not a valid depth.'
        with self.assertRaisesRegexp(IndexError, expected):
            tileset.get_frame(99)

    # PROPERTIES----------------------------------------------------------------
    def test_data(self):
        data = self.get_data()
        tileset = Tileset(data)
        expected = tileset._data
        result = tileset.data
        self.assertIsNot(result, expected)
        self.assertIs(result.content[0], expected.content[0])

        result = result.index.tolist()
        expected = expected.index.tolist()
        self.assertEqual(result, expected)

        del data['content']
        tileset = Tileset(data)
        expected = tileset._data
        result = tileset.data
        self.assertIsNot(result, expected)

    def test_shape(self):
        data = self.get_data()
        result = Tileset(data).shape
        w, h, c = self.get_padded_image().shape
        self.assertEqual(result, (w, h, 2, c))

    def test_shape_in_tiles(self):
        data = self.get_data()
        result = Tileset(data).shape_in_tiles
        self.assertEqual(result, (3, 4, 2))

    def test_width(self):
        data = self.get_data()
        result = Tileset(data).width
        expected = self.get_padded_image().width
        self.assertEqual(result, expected)

    def test_height(self):
        data = self.get_data()
        result = Tileset(data).height
        expected = self.get_padded_image().height
        self.assertEqual(result, expected)

    def test_width_in_tiles(self):
        data = self.get_data()
        result = Tileset(data).width_in_tiles
        self.assertEqual(result, 3)

    def test_height_in_tiles(self):
        data = self.get_data()
        result = Tileset(data).height_in_tiles
        self.assertEqual(result, 4)

    def test_tile_width(self):
        data = self.get_data()
        result = Tileset(data).tile_width
        self.assertEqual(result, 64)

    def test_tile_height(self):
        data = self.get_data()
        result = Tileset(data).tile_height
        self.assertEqual(result, 64)

    def test_tile_shape(self):
        data = self.get_data()
        result = Tileset(data).tile_shape
        self.assertEqual(result, (64, 64, 3))

    def test_num_channels(self):
        data = self.get_data()
        result = Tileset(data).num_channels
        self.assertEqual(result, 3)

    def test_coordinates(self):
        swatch = imt.get_swatch((4, 5, 3), BasicColor.GREEN)
        data = {}
        for w, h, d in product(range(3), range(4), range(2)):
            data[(w, h, d)] = swatch

        result = Tileset.from_dict(data).coordinates
        expected = list(data.keys())
        self.assertEqual(result, expected)

    def test_apply(self):
        def func(coord, image):
            w, h, c = image.shape
            if coord == (2, 1):
                return coord, imt.get_swatch((w, h, c), BasicColor.RED)
            return coord, imt.get_swatch((w, h, c), BasicColor.BLUE)

        data = self.get_data()
        tileset = Tileset(data)
        result = tileset.apply(func)

        shape = result.tile_shape
        red = imt.get_swatch(shape, BasicColor.RED)
        blue = imt.get_swatch(shape, BasicColor.BLUE)

        for coord in tileset.coordinates:
            if coord == (2, 1):
                EnforceImageContent(result.get_tile(coord), '==', red)
            else:
                EnforceImageContent(result.get_tile(coord), '==', blue)
