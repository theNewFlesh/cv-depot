from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lunchbox.enforce import EnforceError
from openexr_tools.enum import ImageCodec
import numpy as np
import openexr_tools.tools as exrtools
import PIL.Image as pil
import pytest

from cv_depot.core.enum import BitDepth, ImageFormat
from cv_depot.core.image import Image
import cv_depot.core.image as cvimg
# ------------------------------------------------------------------------------


class ImageTests(unittest.TestCase):
    def test_has_super_darks(self):
        with self.assertRaises(EnforceError):
            cvimg._has_super_darks('taco')

        img = Image.from_array(np.zeros((10, 10, 3), dtype=np.float32))
        result = cvimg._has_super_darks(img)
        self.assertFalse(result)

        img.data[0, 0, 0] = -0.01
        result = cvimg._has_super_darks(img)
        self.assertTrue(result)

    def test_has_super_brights(self):
        with self.assertRaises(EnforceError):
            cvimg._has_super_brights('pizza')

        img = Image.from_array(np.zeros((10, 10, 3), dtype=np.float32))
        result = cvimg._has_super_brights(img)
        self.assertFalse(result)

        img.data[0, 0, 0] = 1.5
        result = cvimg._has_super_brights(img)
        self.assertTrue(result)

    def test_init(self):
        expected = "Please call one of Image's static constructors to create an"
        expected += ' instance.'
        with self.assertRaisesRegex(NotImplementedError, expected):
            Image(np.zeros((10, 10)))

        expected = r'Illegal number of dimensions for image data. 1 not in \[2, 3\].'
        with self.assertRaisesRegex(AttributeError, expected):
            Image(np.zeros((10)), allow=True)

        expected = r'Illegal number of dimensions for image data. 4 not in \[2, 3\].'
        with self.assertRaisesRegex(AttributeError, expected):
            Image(np.zeros((10, 10, 10, 10)), allow=True)

    def test_from_array(self):
        with self.assertRaises(AttributeError):
            Image.from_array(np.zeros(10, dtype=np.uint8))

        with self.assertRaises(TypeError):
            image = np.zeros((10, 10), dtype=np.float64)
            Image.from_array(image)

        expected = np.zeros((10, 10), dtype=np.float16)
        result = Image.from_array(expected)
        self.assertEqual(
            result.data.ravel().tobytes(),
            expected.ravel().tobytes(),
        )

    def test_read_exr(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.exr')
            image = np.zeros((10, 5, 7), dtype=np.float32)
            expected = np.ones((10, 5), dtype=np.float32)
            image[:, :, 3] = expected
            exrtools.write_exr(filepath, image, {'foo': 'bar'})

            result = Image.read(filepath)
            self.assertEqual(result.format, ImageFormat.EXR)
            self.assertEqual(result.bit_depth, BitDepth.FLOAT32)
            self.assertEqual(result.metadata['foo'], 'bar')
            self.assertEqual(result.num_channels, 7)
            self.assertEqual(result.width, 5)
            self.assertEqual(result.height, 10)
            self.assertEqual(result.data[:, :, 3].tobytes(),
                             expected.tobytes())

    def test_read_png(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.png')
            image = np.zeros((10, 5, 4), dtype=np.uint8)
            expected = np.ones((10, 5), dtype=np.uint8) * 128
            image[:, :, 2] = expected
            pil.fromarray(image).save(filepath)

            result = Image.read(filepath)
            self.assertEqual(result.format, ImageFormat.PNG)
            self.assertEqual(result.bit_depth, BitDepth.UINT8)
            self.assertEqual(result.num_channels, 4)
            self.assertEqual(result.width, 5)
            self.assertEqual(result.height, 10)
            self.assertEqual(result.data[:, :, 2].tobytes(), expected.tobytes())

    def test_read_error(self):
        expected = '/a/bad/path/image.file does not exist.'
        with self.assertRaisesRegex(FileNotFoundError, expected):
            Image.read('/a/bad/path/image.file')

        expected = 'Object of type int is not a str or Path.'
        with self.assertRaisesRegex(TypeError, expected):
            Image.read(999)

    def test_write_error(self):
        image = np.zeros((10, 5, 3), dtype=np.float32)
        expected = 'PNG cannot be written with FLOAT32 data.'
        with self.assertRaisesRegex(TypeError, expected):
            Image.from_array(image).write('/bad/path/test.png')

        image = np.zeros((10, 5, 5), dtype=np.uint8)
        expected = 'JPEG cannot be written with 5 channels. Max channels '
        expected += 'supported: 3.'
        with self.assertRaisesRegex(AttributeError, expected):
            Image.from_array(image).write('/bad/path/test.jpeg')

    def test_write_exr(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.exr')
            expected = np.zeros((10, 5, 7), dtype=np.float32)
            expected[:, :, 3] = np.ones((10, 5), dtype=np.float32)
            meta = dict(taco='pizza', channels=list('rgbaxyz'))
            image = Image.from_array(expected)
            image.metadata = meta
            image.write(filepath)

            result = Image.read(filepath)
            self.assertEqual(
                result.data.ravel().tobytes(),
                expected.ravel().tobytes(),
            )
            self.assertEqual(result.metadata['taco'], 'pizza')
            self.assertEqual(result.channels, meta['channels'])

            expected = ImageCodec.DWAB
            image.write(filepath, codec=expected)
            result = Image.read(filepath).metadata['compression'].name
            self.assertEqual(result, expected.name)

    def test_write_png(self):
        with TemporaryDirectory() as root:
            expected = np.zeros((10, 5, 3), dtype=np.uint8)
            expected[:, :, 0] = np.ones((10, 5), dtype=np.uint8) * 128

            filepath = Path(root, 'test.png')
            Image.from_array(expected).write(filepath)

            result = Image.read(filepath)
            self.assertEqual(
                result.data.ravel().tobytes(),
                expected.ravel().tobytes(),
            )

    def test_write_tiff(self):
        with TemporaryDirectory() as root:
            expected = np.zeros((10, 5, 3), dtype=np.uint8)
            expected[:, :, 0] = np.ones((10, 5), dtype=np.uint8) * 128

            filepath = Path(root, 'test.tiff')
            Image.from_array(expected).write(filepath)

            result = Image.read(filepath)
            self.assertEqual(
                result.data.ravel().tobytes(),
                expected.ravel().tobytes(),
            )

    def test_repr(self):
        with TemporaryDirectory() as root:
            expected = np.zeros((10, 5, 3), dtype=np.uint8)
            expected[:, :, 0] = np.ones((10, 5), dtype=np.uint8) * 128

            filepath = Path(root, 'test.tiff')
            Image.from_array(expected).write(filepath)

            result = Image.read(filepath)._repr()
            expected = '''
       WIDTH: 5
      HEIGHT: 10
NUM_CHANNELS: 3
   BIT_DEPTH: UINT8
      FORMAT: TIFF'''[1:]
            self.assertEqual(result, expected)

    def test_getitem_error(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        expected = np.ones((10, 5), dtype=np.float32)
        temp[:, :, 2] = expected
        img = Image.from_array(temp)

        expected_ = 'Number of dimensions provided: 4, is greater than 3.'
        with self.assertRaisesRegex(IndexError, expected_):
            img[:, :, :, :]

        expected_ = 'foo is not a legal channel name.'
        with self.assertRaisesRegex(IndexError, expected_):
            img[:, :, 'foo']

        expected_ = 'Three lists are not acceptable as indices.'
        with self.assertRaisesRegex(IndexError, expected_):
            ind = [0, 1, 2]
            img[ind, ind, ind]

    def test_string_to_channels(self):
        temp = np.zeros((10, 5, 14), dtype=np.float16)
        img = Image.from_array(temp)
        img = img.set_channels([
            'r', 'g', 'b', 'a',
            'diffuse.r', 'diffuse.g', 'diffuse.b', 'diffuse.a',
            'spec.r', 'spec.g', 'spec.b',
            'x', 'y', 'z',
        ])

        # rgba
        for item in ['r', 'g', 'b', 'a', 'rgb', 'rgba', 'rg', 'agb']:
            result = img._string_to_channels(item)
            self.assertEqual(result, list(item))

        # layer name
        result = img._string_to_channels('diffuse')
        expected = ['diffuse.r', 'diffuse.g', 'diffuse.b', 'diffuse.a']
        self.assertEqual(result, expected)

        result = img._string_to_channels('spec')
        expected = ['spec.r', 'spec.g', 'spec.b']
        self.assertEqual(result, expected)

        # headless layer name
        result = img._string_to_channels('xyz')
        self.assertEqual(result, list('xyz'))

        # non layer
        result = img._string_to_channels('taco')
        self.assertEqual(result, ['taco'])

    def test_getitem(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        expected = np.ones((10, 5), dtype=np.float32)
        temp[:, :, 2] = expected

        img = Image.from_array(temp)
        img = img.set_channels(list('xyz'))

        result = img[:, :, 'z']
        self.assertIsInstance(result, Image)

        expected = expected.ravel().tobytes()
        result = img[:, :, 'z'].data.ravel().tobytes()
        self.assertEqual(result, expected)

        result = img[:, :, [2]].data.ravel().tobytes()
        self.assertEqual(result, expected)

        result = img[:, :, ['z']].data.ravel().tobytes()
        self.assertEqual(result, expected)

        result = img[:, :, 2].data.ravel().tobytes()
        self.assertEqual(result, expected)

        result = img[:, :, 2:3].data.ravel().tobytes()
        self.assertEqual(result, expected)

        expected = img.data[10:20, 0:10, [2, 0]].ravel().tobytes()
        result = img[0:10, 10:20, list('zx')].data.ravel().tobytes()
        self.assertEqual(result, expected)

        expected = img.data[:, :, [2, 0]].ravel().tobytes()
        result = img[:, :, list('zx')].data.ravel().tobytes()
        self.assertEqual(result, expected)

        expected = img.data.ravel().tobytes()
        result = img[:, :, list('xyz')].data.ravel().tobytes()
        self.assertEqual(result, expected)

        foo = [0, 1, 4]
        expected = img.data[:, foo, [2, 1, 0]].ravel().tobytes()
        result = img[foo, :, list('zyx')].data.ravel().tobytes()
        self.assertEqual(result, expected)

        foo = [0, 1, 3]
        expected = img.data[foo, :, [2, 1, 0]].ravel().tobytes()
        result = img[:, foo, list('zyx')].data.ravel().tobytes()
        self.assertEqual(result, expected)

    def test_getitem_metadata(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        img = Image.from_array(temp)
        img = img.set_channels(list('xyz'))
        img.metadata['foo'] = 'bar'
        expected = deepcopy(img.metadata)

        result = img[:, :, :]
        self.assertEqual(result.metadata, expected)

        result = img[:, :, 'xyz']
        self.assertEqual(result.metadata, expected)

        expected['channels'] = ['x']
        result = img[:, :, 'x']
        self.assertEqual(result.metadata, expected)

        expected['channels'] = ['x', 'z']
        result = img[:, :, list('xz')]
        self.assertEqual(result.metadata, expected)

        result = img[:, :, 1:3]
        expected['channels'] = ['y', 'z']
        self.assertEqual(result.metadata, expected)

        img = img.set_channels(['x.0', 'x.1', 'x.2'])
        expected = deepcopy(img.metadata)

        result = img[:, :, 'x']
        self.assertEqual(result.metadata, expected)

    def test_getitem_variable_indices(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        expected = np.ones((10, 5), dtype=np.float32)
        temp[:, :, 2] = expected

        img = Image.from_array(temp)
        img = img.set_channels(list('xyz'))

        expected = temp[:, :10].ravel().tobytes()
        result = img[:10].data.ravel().tobytes()
        self.assertEqual(result, expected)

        expected = temp[1:8, :10].ravel().tobytes()
        result = img[:10, 1:8].data.ravel().tobytes()
        self.assertEqual(result, expected)

    def test_getitem_tuple(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        expected = np.ones((10, 5), dtype=np.float32)
        temp[:, :, 2] = expected

        img = Image.from_array(temp)
        img = img.set_channels(list('xyz'))
        foo = (0, 1, 2)

        expected = img.data[:, foo, :].ravel().tobytes()
        result = img[foo, :, :].data.ravel().tobytes()
        self.assertEqual(result, expected)

        expected = img.data[foo, :, :].ravel().tobytes()
        result = img[:, foo, :].data.ravel().tobytes()
        self.assertEqual(result, expected)

    def test_to_bit_depth_float(self):
        img = np.ones((10, 10, 3), dtype=np.float16)
        result = Image.from_array(img).to_bit_depth(BitDepth.FLOAT16)\
            .data.ravel().tobytes()
        self.assertEqual(result, img.ravel().tobytes())

        bds = [
            BitDepth.INT8, BitDepth.UINT8, BitDepth.FLOAT16, BitDepth.FLOAT32
        ]
        for bd in bds:
            img = np.ones((10, 10, 3), dtype=np.float16)
            result = Image.from_array(img).to_bit_depth(bd)
            self.assertEqual(result.bit_depth, bd)

        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = np.ones((10, 10), dtype=np.float32)
        result = Image.from_array(img).to_bit_depth(BitDepth.UINT8)
        self.assertEqual(result.data.min(), 0)
        self.assertEqual(result.data.max(), 255)

        bds = [BitDepth.FLOAT16, BitDepth.FLOAT32]
        for bd in bds:
            img = np.zeros((10, 10, 3), dtype=np.float16)
            img[:, :, 0] = np.ones((10, 10), dtype=np.float16)
            result = Image.from_array(img).to_bit_depth(bd)
            self.assertEqual(result.data.min(), 0)
            self.assertEqual(result.data.max(), 1.0)

    def test_to_bit_depth_int(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = Image.from_array(img).to_bit_depth(BitDepth.UINT8)\
            .data.ravel().tobytes()
        self.assertEqual(result, img.ravel().tobytes())

        bds = [
            BitDepth.INT8, BitDepth.UINT8, BitDepth.FLOAT16, BitDepth.FLOAT32
        ]
        for bd in bds:
            img = np.ones((10, 10, 3), dtype=np.uint8) * 255
            result = Image.from_array(img).to_bit_depth(bd)
            self.assertEqual(result.bit_depth, bd)

        bds = [BitDepth.FLOAT16, BitDepth.FLOAT32]
        for bd in bds:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            img[:, :, 0] = np.ones((10, 10), dtype=np.uint8) * 255
            result = Image.from_array(img).to_bit_depth(bd)
            self.assertEqual(result.data.min(), 0)
            self.assertEqual(result.data.max(), 1.0)

    def test_to_bit_depth_supers(self):
        img = np.zeros((10, 10, 3), dtype=np.float16)
        img[:, :, 0] = np.ones((10, 10), dtype=np.float16) + 0.2

        expected = r'Image has values above 1. Max value: 1.2\d?'
        with self.assertRaisesRegex(ValueError, expected):
            Image.from_array(img).to_bit_depth(BitDepth.UINT8)

        img[:, :, 0] = np.zeros((10, 10), dtype=np.float16) - 0.1
        expected = r'Image has values below 0. Min value: -0.\d?'
        with self.assertRaisesRegex(ValueError, expected):
            Image.from_array(img).to_bit_depth(BitDepth.UINT8)

    def test_to_unit_space(self):
        img = np.zeros((10, 10, 3), dtype=np.float16)
        img[:, :, 0] = np.zeros((10, 10), dtype=np.float16) - 0.9
        img[:, :, 1] = np.zeros((10, 10), dtype=np.float16) + 5.2

        image = cvimg.Image.from_array(img)
        self.assertEqual(image.data.min(), -0.9)
        self.assertEqual(image.data.max(), 5.2)

        result = image.to_unit_space()
        self.assertEqual(result.data.min(), 0)
        self.assertEqual(result.data.max(), 1.0)

    def test_compare(self):
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.float16)
        img1 = Image.from_array(img1)
        img2 = Image.from_array(img2)

        result = img1.compare(img2)
        expected = {
            'bit_depth': ('UINT8', 'FLOAT16'),
            'bits': (8, 16),
            'channels': (['r', 'g', 'b'], ['r', 'g', 'b']),
            'dtype': (np.uint8, np.float16),
            'format_bit_depths': (None, None),
            'format_channels': (None, None),
            'format_custom_metadata': (None, None),
            'format_extension': (None, None),
            'format_max_channels': (None, None),
            'height': (10, 10),
            'num_channels': (3, 3),
            'signed': (False, True),
            'type': (int, float),
            'width': (10, 10),
        }
        self.assertEqual(result, expected)

    def test_compare_content(self):
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8)
        img1 = Image.from_array(img1)
        img2 = Image.from_array(img2)

        result = img1.compare(img1, content=True, diff_only=True)
        self.assertEqual(result, {})

        result = img1.compare(img2, diff_only=True)
        self.assertEqual(result, {})

        result = img1.compare(img2, content=True, diff_only=True)
        self.assertEqual(list(result.keys()), ['mean_content_difference'])

        result = result['mean_content_difference']
        self.assertAlmostEqual(result, 1 / 256, delta=0.001)

    def test_compare_content_errors(self):
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((5, 5, 3), dtype=np.uint8)
        img1 = Image.from_array(img1)
        img2 = Image.from_array(img2)

        expected = 'Cannot compare images'
        with self.assertRaisesRegex(ValueError, expected):
            img1.compare(img2, content=True)

    def test_compare_diff(self):
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.zeros((10, 10, 3), dtype=np.float16)
        img1 = Image.from_array(img1)
        img2 = Image.from_array(img2)

        result = img1.compare(img2, diff_only=True)
        expected = {
            'bit_depth': ('UINT8', 'FLOAT16'),
            'bits': (8, 16),
            'dtype': (np.uint8, np.float16),
            'signed': (False, True),
            'type': (int, float),
        }
        self.assertEqual(result, expected)

        result = img1.compare(img1, diff_only=True)
        self.assertEqual(result, {})

    def test_eq(self):
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img1 = Image.from_array(img1)
        self.assertTrue(img1 == img1)

        img2 = np.zeros((10, 10, 3), dtype=np.int8)
        img2 = Image.from_array(img2)
        self.assertFalse(img1 == img2)

        img2 = np.ones((10, 10, 3), dtype=np.uint8)
        img2 = Image.from_array(img2)
        self.assertFalse(img1 == img2)

    def test_info(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.exr')
            expected = np.zeros((10, 5, 7), dtype=np.float32)
            expected[:, :, 3] = np.ones((10, 5), dtype=np.float32)
            meta = dict(foo='bar', channels=list('rgbaxyz'))
            image = Image.from_array(expected)
            image.metadata = meta
            image.write(filepath)

            result = Image.read(filepath)
            expected = dict(
                width=5,
                height=10,
                num_channels=7,
                type=float,
                bits=32,
                bit_depth='FLOAT32',
                channels=['r', 'g', 'b', 'a', 'x', 'y', 'z'],
                dtype=np.float32,
                signed=True,
                format_bit_depths=[BitDepth.FLOAT16, BitDepth.FLOAT32],
                format_channels=['r', 'g', 'b', 'a', '...'],
                format_custom_metadata=True,
                format_extension='exr',
                format_max_channels=1023,
            )
            self.assertEqual(result.info, expected)

    def test_set_channels(self):
        img = np.zeros((10, 10, 5), dtype=np.float16)

        expected = list('abcde')
        result = Image\
            .from_array(img)\
            .set_channels(expected)\
            .channels
        self.assertEqual(result, expected)

        expected = 'Number of channels given does not equal last dimension size'
        expected = '. 3 != 5.'
        with self.assertRaisesRegex(ValueError, expected):
            Image.from_array(img).set_channels(list('abc'))

        with pytest.raises(ValueError) as e:
            Image.from_array(img).set_channels(list('aabbc'))
        expected = "Duplicate channel names found: ['a', 'b']."
        self.assertEqual(str(e.value), expected)

    def test_extension(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.exr')
            expected = np.zeros((10, 5, 7), dtype=np.float32)
            image = Image.from_array(expected)

            self.assertEqual(image.extension, None)

            image.write(filepath)
            result = Image.read(filepath)
            self.assertEqual(result.extension, ImageFormat.EXR.extension)

    def test_shape(self):
        temp = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(temp)
        self.assertEqual(result.shape, (5, 10, 7))

        temp = np.zeros((10, 5), dtype=np.uint8)
        result = Image.from_array(temp)
        self.assertEqual(result.shape, (5, 10, 1))

    def test_width(self):
        expected = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(expected)
        self.assertEqual(result.width, expected.shape[1])

    def test_height(self):
        expected = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(expected)
        self.assertEqual(result.height, expected.shape[0])

    def test_width_and_height(self):
        temp = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(temp)
        self.assertEqual(result.width_and_height, (5, 10))

    def test_channels(self):
        temp = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(temp)
        expected = list('rgba') + [4, 5, 6]
        result.set_channels(expected)
        self.assertEqual(result.channels, expected)

        temp = np.zeros((10, 5), dtype=np.float32)
        result = Image.from_array(temp)
        self.assertEqual(result.channels, ['l'])

        temp = np.zeros((10, 5, 6), dtype=np.float32)
        result = Image.from_array(temp)
        expected = list('rgba') + [4, 5]
        self.assertEqual(result.channels, expected)

    def test_max_channels(self):
        with TemporaryDirectory() as root:
            filepath = Path(root, 'test.exr')
            expected = np.zeros((10, 5, 7), dtype=np.float32)
            image = Image.from_array(expected)

            self.assertEqual(image.max_channels, None)

            image.write(filepath)
            result = Image.read(filepath)
            self.assertEqual(result.max_channels, ImageFormat.EXR.max_channels)

    def test_channel_layers(self):
        arr = np.zeros((10, 5, 11), dtype=np.float32)
        image = Image.from_array(arr)

        expected = ['rgba', '4567', '8910']
        self.assertEqual(image.channel_layers, expected)

        image = image.set_channels([
            'r', 'g', 'b',
            'diffuse.r', 'diffuse.g', 'diffuse.b', 'diffuse.a',
            'spec.r', 'spec.g', 'spec.b',
            'depth.z',
        ])
        expected = ['rgb', 'diffuse', 'spec', 'depth']
        self.assertEqual(image.channel_layers, expected)

    def test_num_channels(self):
        temp = np.zeros((10, 5, 7), dtype=np.float32)
        result = Image.from_array(temp)
        self.assertEqual(result.num_channels, 7)

        temp = np.zeros((10, 5), dtype=np.float16)
        result = Image.from_array(temp)
        self.assertEqual(result.num_channels, 1)

    def test_bit_depth(self):
        temp = np.zeros((10, 5, 3), dtype=np.float32)
        result = Image.from_array(temp)
        result._data = result.data.astype(np.float16)
        self.assertEqual(result.bit_depth, BitDepth.FLOAT16)

    def test_to_pil(self):
        # 1 channel
        temp = np.zeros((10, 5), dtype=np.uint8)
        result = Image.from_array(temp).to_pil()
        self.assertIsInstance(result, pil.Image)

        # 1 channel
        temp = np.zeros((10, 5, 1), dtype=np.int8)
        result = Image.from_array(temp).to_pil()
        self.assertIsInstance(result, pil.Image)

        # 3 channels
        temp = np.zeros((10, 5, 3), dtype=np.float16)
        result = Image.from_array(temp).to_pil()
        self.assertIsInstance(result, pil.Image)

        # 3 channels
        temp = np.zeros((10, 5, 4), dtype=np.float32)
        result = Image.from_array(temp).to_pil()
        self.assertIsInstance(result, pil.Image)

    def test_to_pil_errors(self):
        expected = 'PIL only accepts image with 1, 3 or 4 channels.'
        temp = np.zeros((10, 5, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, expected):
            Image.from_array(temp).to_pil()
