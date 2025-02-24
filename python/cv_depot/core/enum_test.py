import unittest

from lunchbox.enforce import EnforceError
import numpy as np

from cv_depot.core.enum import EnumBase, BitDepth, ImageFormat, VideoFormat, VideoCodec
# ------------------------------------------------------------------------------


class FakeEnum(EnumBase):
    FOO_BAR = 1
    TACO = 2


class EnumBaseTests(unittest.TestCase):
    def test_repr(self):
        self.assertEqual(repr(FakeEnum.FOO_BAR), 'FakeEnum.FOO_BAR')

    def test_from_string(self):
        for item in ['foo_bar', 'FOO_BAR', 'foo-bar', 'FOO-BAR', 'Foo-Bar']:
            result = FakeEnum.from_string(item)
            self.assertEqual(result, FakeEnum.FOO_BAR)

        result = FakeEnum.from_string('taco')
        self.assertEqual(result, FakeEnum.TACO)

    def test_from_string_errors(self):
        expected = 'Value given is not a string. 99 !=.*str'
        with self.assertRaisesRegex(EnforceError, expected):
            FakeEnum.from_string(99)

        expected = 'FOO is not a FakeEnum option. Options:'
        with self.assertRaisesRegex(EnforceError, expected):
            FakeEnum.from_string('foo')
# ------------------------------------------------------------------------------


class BitDepthTests(unittest.TestCase):
    def test_repr(self):
        b = BitDepth.FLOAT32
        result = repr(b)
        expected = 'BitDepth.FLOAT32'
        self.assertEqual(result, expected)

    def test_from_dtype(self):
        result = BitDepth.from_dtype(np.float16)
        self.assertEqual(result, BitDepth.FLOAT16)

        result = BitDepth.from_dtype(np.float32)
        self.assertEqual(result, BitDepth.FLOAT32)

        result = BitDepth.from_dtype(np.int8)
        self.assertEqual(result, BitDepth.INT8)

        result = BitDepth.from_dtype(np.uint8)
        self.assertEqual(result, BitDepth.UINT8)

        expected = 'float64 is not a supported bit depth.'
        with self.assertRaisesRegex(TypeError, expected):
            BitDepth.from_dtype(np.float64)
# ------------------------------------------------------------------------------


class ImageFormatTests(unittest.TestCase):
    def test_repr(self):
        fmat = ImageFormat.EXR
        result = repr(fmat)
        expected = '''
<ImageFormat.EXR>
      extension: exr
     bit_depths: ['FLOAT16', 'FLOAT32']
       channels: ['r', 'g', 'b', 'a', '...']
   max_channels: 1023
custom_metadata: True'''[1:]
        self.assertEqual(result, expected)

    def test_from_extension(self):
        # exr
        exrs = ['exr', 'EXR', '.Exr', '.eXr', '.exr', '.EXR']
        for exr in exrs:
            result = ImageFormat.from_extension(exr)
            self.assertEqual(result, ImageFormat.EXR)

        # png
        pngs = ['png', 'Png', 'PNG', '.png', '.PNG', '.Png', 'pNg']
        for png in pngs:
            result = ImageFormat.from_extension(png)
            self.assertEqual(result, ImageFormat.PNG)

        # jpeg
        jpegs = [
            'jpg', 'jpeg', '.jpg', '.jPeG', '.jpeg', '.Jpeg', 'Jpeg', 'JPEG',
        ]
        for jpg in jpegs:
            result = ImageFormat.from_extension(jpg)
            self.assertEqual(result, ImageFormat.JPEG)

        # tiff
        tiffs = [
            'tif', '.Tiff', 'tiff', 'Tiff', '.TIFF', 'TIFF', '.tif', '.tiff',
            '.tIFF'
        ]
        for tif in tiffs:
            result = ImageFormat.from_extension(tif)
            self.assertEqual(result, ImageFormat.TIFF)

        # other
        expected = 'ImageFormat not found for given extension: .foo'
        with self.assertRaisesRegex(TypeError, expected):
            ImageFormat.from_extension('.foo')
# ------------------------------------------------------------------------------


class VideoCodecTests(unittest.TestCase):
    def test_repr(self):
        x = VideoCodec.H265
        result = repr(x)
        expected = '''
<VideoCodec.H265>
  string: h265
  ffmpeg_code: hevc'''[1:]
        self.assertEqual(result, expected)


class VideoFormatTests(unittest.TestCase):
    def test_repr(self):
        fmat = VideoFormat.MP4
        result = repr(fmat)
        expected = '''
<VideoFormat.MP4>
      extension: mp4
     bit_depths: ['UINT8']
       channels: ['r', 'g', 'b']
   max_channels: 3
custom_metadata: False'''[1:]
        self.assertEqual(result, expected)

    def test_from_extension(self):
        m4vs = ['M4v', 'M4V', 'm4v', '.m4v', 'm4V', '.M4V', '.M4v']
        for m4v in m4vs:
            result = VideoFormat.from_extension(m4v)
            self.assertEqual(result, VideoFormat.M4V)

        result = VideoFormat.from_extension('mpg')
        self.assertEqual(result, VideoFormat.MPEG)

        result = VideoFormat.from_extension('mpeg')
        self.assertEqual(result, VideoFormat.MPEG)

        result = VideoFormat.from_extension('mp4')
        self.assertEqual(result, VideoFormat.MP4)

        result = VideoFormat.from_extension('mov')
        self.assertEqual(result, VideoFormat.MOV)

        expected = 'VideoFormat not found for given extension: .foo'
        with self.assertRaisesRegex(TypeError, expected):
            VideoFormat.from_extension('.foo')
