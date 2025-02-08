import unittest

from lunchbox.enforce import EnforceError
import cv2
import numpy as np

from cv_depot.core.color import BasicColor
from cv_depot.core.enum import BitDepth
from cv_depot.core.image import Image
import cv_depot.ops.draw as cvdraw
import cv_depot.ops.edit as cvedit
import cv_depot.ops.filter as cvfilt
# ------------------------------------------------------------------------------


class FilterTests(unittest.TestCase):
    def test_canny_edges(self):
        img = cvdraw.checkerboard(2, 1, (50, 100))
        result = cvfilt.canny_edges(img)
        self.assertEqual(result.data[0, 0], 0)
        self.assertEqual(result.data[0, 99], 0)
        self.assertEqual(result.data[99, 0], 0)
        self.assertEqual(result.data[99, 99], 0)
        self.assertEqual(result.data[0, 49], 1)
        self.assertEqual(result.data[49, 49], 1)
        self.assertEqual(result.data[99, 49], 1)

    def test_canny_edges_size(self):
        img = cvdraw.checkerboard(2, 1, (50, 100)).data
        img = np.concatenate([img, img, img], axis=2)
        img = Image.from_array(img)
        result = cvfilt.canny_edges(img, size=4)

        self.assertEqual(result.data[0, 0], 0)
        self.assertEqual(result.data[0, 99], 0)
        self.assertEqual(result.data[99, 0], 0)
        self.assertEqual(result.data[99, 99], 0)

        self.assertEqual(result.data[0, 44:55].sum(), 9)
        self.assertEqual(result.data[48, 44:55].sum(), 9)
        self.assertEqual(result.data[99, 44:55].sum(), 9)

    def test_canny_edges_errors(self):
        # image
        with self.assertRaises(EnforceError):
            cvfilt.canny_edges('foo')

        # size float
        img = cvdraw.checkerboard(2, 1, (50, 100))
        with self.assertRaises(EnforceError):
            cvfilt.canny_edges(img, 1.0)

        # size < 0
        with self.assertRaises(EnforceError):
            cvfilt.canny_edges(img, -1)

    def get_tophat_image(self):
        img = np.zeros((128, 128, 3))
        img = cv2.circle(img, (64, 64), 30, (255), -1)
        img = cv2.circle(img, (100, 100), 10, (255), -1)
        img = img.astype(np.uint8)
        img = Image.from_array(img)
        return img

    def test_tophat_close(self):
        img = self.get_tophat_image()
        before = img.data.sum()
        result = cvfilt.tophat(img, 10, kind='close').data.sum()
        self.assertGreater(before, result)

    def test_tophat_open(self):
        img = self.get_tophat_image()
        before = img.data.sum()
        result = cvfilt.tophat(img, 25, kind='open').data.sum()
        self.assertLess(before, result)

    def test_tophat_error(self):
        # image
        with self.assertRaises(EnforceError):
            cvfilt.tophat(np.zeros((10, 10, 3)), 10)

        # amount
        swatch = cvdraw.swatch((10, 10, 3), BasicColor.BLACK)
        with self.assertRaises(EnforceError):
            cvfilt.tophat(swatch, -1)

        # kind
        expected = r'Illegal tophat kind: foo\. Legal tophat kinds: '
        expected += r"\['open', 'close'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            cvfilt.tophat(swatch, 10, kind='foo')

    def test_linear_lookup(self):
        lut = cvfilt.linear_lookup(0.25, 0.75)

        img = np.ones((10, 10), dtype=np.float32) * 0.8
        result = lut(img)
        self.assertEqual(result.mean(), 1)

        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = lut(img)
        self.assertEqual(result.mean(), 0.5)

        img = np.ones((10, 10), dtype=np.float32) * 0.4
        result = lut(img)
        self.assertEqual(round(result.mean(), 2), 0.3)

        img = np.ones((10, 10), dtype=np.float32) * 0.2
        result = lut(img)
        self.assertEqual(result.mean(), 0)

    def test_linear_smooth(self):
        img = np.zeros((256, 256))
        img = cv2 \
            .rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1) \
            .astype(np.uint8)

        bit_depth = BitDepth.FLOAT16
        img1 = Image.from_array(img).to_bit_depth(bit_depth)

        img2 = cvfilt.linear_smooth(img1, blur=40, lower=0.45, upper=0.55)
        self.assertEqual(img2.bit_depth, bit_depth)

        img1 = img1.data
        img2 = img2.data
        result = np.minimum(img1, img2).sum()
        self.assertLess(result, img1.sum())

    def test_linear_smooth_error(self):
        # image
        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(np.zeros((10, 10, 3)))

        # blur
        swatch = cvdraw.swatch((10, 10, 3), BasicColor.CYAN)
        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(swatch, blur=-2)

        # lower
        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(swatch, lower=-1)

        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(swatch, lower=2)

        expected = 'Lower bound cannot be greater than upper bound. 0.9 > 0.1'
        with self.assertRaisesRegex(EnforceError, expected):
            cvfilt.linear_smooth(swatch, lower=0.9, upper=0.1)

        # upper
        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(swatch, upper=3)

        with self.assertRaises(EnforceError):
            cvfilt.linear_smooth(swatch, upper=-1)

    def test_exact_color_key(self):
        shape = (10, 10, 3)
        red = cvdraw.swatch(shape, BasicColor.RED)
        blue = cvdraw.swatch(shape, BasicColor.BLUE)
        img = cvedit.staple(red, blue)

        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        expected = cvedit \
            .staple(white, black)[:, :, 0:1] \
            .set_channels(['a'])

        result = cvfilt.key_exact_color(img, BasicColor.RED)
        self.assertEqual(result.channels, list('rgba'))
        self.assertEqual(result[:, :, list('rgb')], img)
        self.assertEqual(result[:, :, 'a'], expected)
        self.assertEqual(result.bit_depth, img.bit_depth)

    def test_exact_color_key_one_channel(self):
        shape = (10, 10, 3)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)

        img = cvedit \
            .staple(white, black)[:, :, 'r'] \
            .set_channels(['x'])

        expected = cvedit \
            .staple(black, white)[:, :, 'r'] \
            .set_channels(['a'])

        result = cvfilt.key_exact_color(img, BasicColor.BLACK)
        self.assertEqual(result.channels, list('xa'))
        self.assertEqual(result[:, :, 'x'], img)
        self.assertEqual(result[:, :, 'a'], expected)

    def test_exact_color_key_invert(self):
        shape = (10, 10, 3)
        red = cvdraw.swatch(shape, BasicColor.RED)
        blue = cvdraw.swatch(shape, BasicColor.BLUE)
        img = cvedit.staple(red, blue).to_bit_depth(BitDepth.UINT8)

        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        expected = cvedit \
            .staple(black, white)[:, :, 'r'] \
            .to_bit_depth(BitDepth.UINT8) \
            .set_channels(['a'])

        result = cvfilt.key_exact_color(img, BasicColor.RED, invert=True)
        self.assertEqual(result.channels, list('rgba'))
        self.assertEqual(result[:, :, 'a'], expected)

    def test_exact_color_key_channel(self):
        shape = (10, 10, 3)
        red = cvdraw.swatch(shape, BasicColor.RED)
        blue = cvdraw.swatch(shape, BasicColor.BLUE)
        img = cvedit.staple(red, blue)

        alpha = cvdraw \
            .swatch(img.shape, BasicColor.BLACK)[:, :, 'r'] \
            .data[..., np.newaxis]
        img = np.concatenate([img.data, alpha], axis=2)
        img = Image.from_array(img)

        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        expected = cvedit \
            .staple(black, white)[:, :, 'r'] \
            .set_channels(['a'])

        result = cvfilt.key_exact_color(img, BasicColor.BLUE, channel='a')
        self.assertEqual(result[:, :, 'a'], expected)

        expected = expected.set_channels(['foo.bar'])
        result = cvfilt.key_exact_color(img, BasicColor.BLUE, channel='foo.bar')
        self.assertEqual(result[:, :, 'foo.bar'], expected)

    def test_exact_color_key_errors(self):
        shape = (10, 10, 3)
        red = cvdraw.swatch(shape, BasicColor.RED)
        blue = cvdraw.swatch(shape, BasicColor.BLUE)
        img = cvedit.staple(red, blue)

        # image
        with self.assertRaises(EnforceError):
            cvfilt.key_exact_color('foobar', BasicColor.RED)

        # channel
        with self.assertRaises(EnforceError):
            cvfilt.key_exact_color(img, BasicColor.RED, channel=99)

        # invert
        with self.assertRaises(EnforceError):
            cvfilt.key_exact_color(img, BasicColor.RED, invert='foobar')

        # two channel
        expected = r"\['b'\] not found in image channels\. "
        expected += r"Given channels: \['r', 'g'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            cvfilt.key_exact_color(img[:, :, list('rg')], BasicColor.RED)
