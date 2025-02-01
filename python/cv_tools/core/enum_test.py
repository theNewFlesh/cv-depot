import unittest

from lunchbox.enforce import EnforceError
import numpy as np

from cv_tools.core.enum import (
    BitDepth, ImageFormat, ImageCodec, VideoFormat, VideoCodec
)
# ------------------------------------------------------------------------------


class BitDepthTests(unittest.TestCase):
    def test_repr(self):
        b = BitDepth.FLOAT32
        result = repr(b)
        expected = '''
<BitDepth.FLOAT32>
 dtype: float32
  bits: 32
signed: True
  type: float'''[1:]
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

    def test_from_string(self):
        result = BitDepth.from_string('float16')
        self.assertEqual(result, BitDepth.FLOAT16)

        result = BitDepth.from_string('float32')
        self.assertEqual(result, BitDepth.FLOAT32)

        result = BitDepth.from_string('int8')
        self.assertEqual(result, BitDepth.INT8)

        result = BitDepth.from_string('uint8')
        self.assertEqual(result, BitDepth.UINT8)

        expected = 'foo is not a supported bit depth.'
        with self.assertRaisesRegex(TypeError, expected):
            BitDepth.from_string('foo')
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


class ImageCodecTests(unittest.TestCase):
    def test_repr(self):
        x = ImageCodec.PXR24
        result = repr(x)
        expected = '''
<ImageCodec.PXR24>
  string: pxr24
exr_code: 5'''[1:]
        self.assertEqual(result, expected)

    def test_from_string(self):
        self.assertEqual(ImageCodec.from_string('piz'), ImageCodec.PIZ)
        self.assertEqual(ImageCodec.from_string('b44'), ImageCodec.B44)
        self.assertEqual(ImageCodec.from_string('b44a'), ImageCodec.B44A)
        self.assertEqual(ImageCodec.from_string('dwaa'), ImageCodec.DWAA)
        self.assertEqual(ImageCodec.from_string('dwab'), ImageCodec.DWAB)
        self.assertEqual(ImageCodec.from_string('pxr24'), ImageCodec.PXR24)
        self.assertEqual(ImageCodec.from_string('rle'), ImageCodec.RLE)
        self.assertEqual(ImageCodec.from_string('zip'), ImageCodec.ZIP)
        self.assertEqual(ImageCodec.from_string('zips'), ImageCodec.ZIPS)
        self.assertEqual(
            ImageCodec.from_string('uncompressed'),
            ImageCodec.UNCOMPRESSED
        )

    def test_from_string_errors(self):
        expected = 'Value given is not a string. 54 !=.*str'
        with self.assertRaisesRegex(EnforceError, expected):
            ImageCodec.from_string(54)

        expected = '"foo" has no legal ImageCodec type. '
        expected += 'Legal codec strings: .*'
        with self.assertRaisesRegex(EnforceError, expected):
            ImageCodec.from_string('foo')

    def test_from_exr_code(self):
        self.assertEqual(ImageCodec.from_exr_code(4), ImageCodec.PIZ)
        self.assertEqual(ImageCodec.from_exr_code(6), ImageCodec.B44)
        self.assertEqual(ImageCodec.from_exr_code(7), ImageCodec.B44A)
        self.assertEqual(ImageCodec.from_exr_code(8), ImageCodec.DWAA)
        self.assertEqual(ImageCodec.from_exr_code(9), ImageCodec.DWAB)
        self.assertEqual(ImageCodec.from_exr_code(5), ImageCodec.PXR24)
        self.assertEqual(ImageCodec.from_exr_code(1), ImageCodec.RLE)
        self.assertEqual(ImageCodec.from_exr_code(3), ImageCodec.ZIP)
        self.assertEqual(ImageCodec.from_exr_code(2), ImageCodec.ZIPS)
        self.assertEqual(
            ImageCodec.from_exr_code(0),
            ImageCodec.UNCOMPRESSED
        )

    def test_from_exr_code_errors(self):
        expected = 'Value given is not an integer. foo !=.*int'
        with self.assertRaisesRegex(EnforceError, expected):
            ImageCodec.from_exr_code('foo')

        expected = 'EXR code 42 has no legal ImageCodec type. '
        expected += 'Legal EXR codes: .*'
        with self.assertRaisesRegex(EnforceError, expected):
            ImageCodec.from_exr_code(42)
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

    def test_from_string(self):
        self.assertEqual(VideoCodec.from_string('H264'), VideoCodec.H264)

    def test_from_string_errors(self):
        expected = 'Value given is not a string. 54 !=.*str'
        with self.assertRaisesRegex(EnforceError, expected):
            VideoCodec.from_string(54)

        expected = '"foo" has no legal VideoCodec type. '
        expected += 'Legal codec strings: .*'
        with self.assertRaisesRegex(EnforceError, expected):
            VideoCodec.from_string('foo')


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
