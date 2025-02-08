import unittest

from lunchbox.enforce import EnforceError
import cv2
import numpy as np

from cv_depot.core.channel_map import ChannelMap
from cv_depot.core.color import BasicColor
from cv_depot.core.image import BitDepth, Image
import cv_depot.ops.channel as cvchan
import cv_depot.ops.draw as cvdraw
# ------------------------------------------------------------------------------


class ChannelTests(unittest.TestCase):
    def get_content_diff(self, a, b):
        return a.compare(b, content=True)['mean_content_difference']

    def assert_similar_content(self, a, b, epsilon=1):
        diff = self.get_content_diff(a, b)
        self.assertLess(diff, 1)

    def assert_equal_content(self, a, b):
        diff = self.get_content_diff(a, b)
        self.assertEqual(diff, 0)

    def test_has_super_brights(self):
        with self.assertRaises(EnforceError):
            cvchan.has_super_brights('foo')

        img = Image.from_array(np.zeros((10, 10, 3), dtype=np.float32))
        result = cvchan.has_super_brights(img)
        self.assertFalse(result)

        img.data[0, 0, 0] = 2.9
        result = cvchan.has_super_brights(img)
        self.assertTrue(result)

    def test_has_super_darks(self):
        with self.assertRaises(EnforceError):
            cvchan.has_super_darks('foo')

        img = Image.from_array(np.zeros((10, 10, 3), dtype=np.float32))
        result = cvchan.has_super_darks(img)
        self.assertFalse(result)

        img.data[0, 0, 0] = -0.09
        result = cvchan.has_super_darks(img)
        self.assertTrue(result)

    def test_to_hsv(self):
        image = np.zeros((256, 256, 3), dtype=np.float32)
        image = cv2.rectangle(
            image, (50, 50), (200, 200), (1, 0.5, 0.25), -1
        )
        image = Image.from_array(image)

        # float
        result = cvchan.to_hsv(image)
        max_ = result[:, :, 'h'].data.max()
        self.assertLessEqual(max_, 1)

        # int
        image2 = image.to_bit_depth(cvchan.BitDepth.UINT8)
        result = cvchan.to_hsv(image2)
        max_ = result[:, :, 'h'].data.max()
        self.assertLessEqual(max_, 255)

        # back to rgb
        result = cvchan.to_hsv(image)
        result = cvchan.to_rgb(result)
        self.assert_similar_content(result, image)

    def test_to_hsv_error(self):
        img = np.zeros((256, 256, 3))
        img = cv2.rectangle(img, (50, 50), (200, 200), (255, 128, 90), -1)
        img = img.astype(np.float32)
        img = Image.from_array(img).set_channels(list('rgx'))

        expected = 'Image does not contain RGB channels. Channels found: '
        expected += r"\['r', 'g', 'x'\]."
        with self.assertRaisesRegex(AttributeError, expected):
            cvchan.to_hsv(img)

    def test_to_rgb(self):
        image = np.zeros((256, 256, 3), dtype=np.float32)
        image = cv2.rectangle(
            image, (50, 50), (200, 200), (1, 0.5, 0.25), -1
        )
        image = Image.from_array(image)
        image = cvchan.to_hsv(image)

        # float
        result = cvchan.to_rgb(image)
        max_ = result[:, :, 'r'].data.max()
        self.assertLessEqual(max_, 1)

        # int
        image2 = image.to_bit_depth(cvchan.BitDepth.UINT8)
        result = cvchan.to_rgb(image2)
        max_ = result[:, :, 'r'].data.max()
        self.assertLessEqual(max_, 255)

        # back to hsv
        result = cvchan.to_rgb(image)
        result = cvchan.to_hsv(result)
        self.assert_similar_content(result, image)

    def test_to_rgb_error(self):
        img = np.zeros((256, 256, 3))
        img = cv2.rectangle(img, (50, 50), (200, 200), (255, 128, 90), -1)
        img = img.astype(np.float32)
        img = Image.from_array(img).set_channels(list('hsx'))

        expected = 'Image does not contain HSV channels. Channels found: '
        expected += r"\['h', 's', 'x'\]."
        with self.assertRaisesRegex(AttributeError, expected):
            cvchan.to_rgb(img)

    def test_mix(self):
        shape = (10, 10, 1)
        a = cvdraw.swatch(shape, BasicColor.BLACK)
        b = cvdraw.swatch(shape, BasicColor.WHITE)
        result = cvchan.mix(a, b, amount=0.5)
        expected = cvdraw.swatch(shape, BasicColor.GREY)

        # 0
        result = cvchan.mix(a, b, amount=0)
        self.assertEqual(result, b)

        # 0.5
        result = cvchan.mix(a, b, amount=0.5)
        expected = cvdraw.swatch(shape, BasicColor.GREY)
        self.assertEqual(result, expected)

        # 1
        result = cvchan.mix(a, b, amount=1)
        self.assertEqual(result, a)

        # bit_depth
        result = cvchan.mix(a, b, amount=0.5)
        self.assertEqual(result.bit_depth, a.bit_depth)

    def test_mix_errors(self):
        shape = (10, 10, 1)
        a = cvdraw.swatch(shape, BasicColor.BLACK)

        with self.assertRaises(EnforceError):
            cvchan.mix('foo', a, amount=0.5)

        with self.assertRaises(EnforceError):
            cvchan.mix(a, 'foo', amount=0.5)

        with self.assertRaises(EnforceError):
            cvchan.mix(a, a, amount=-0.1)

        with self.assertRaises(EnforceError):
            cvchan.mix(a, a, amount=1.2)

    def test_invert(self):
        img = cvdraw.checkerboard(2, 2, (50, 50))
        expected = cvdraw.checkerboard(3, 2, (50, 50))[50:]
        result = cvchan.invert(img)
        self.assertEqual(result, expected)

        # 8 bit
        img = img.to_bit_depth(BitDepth.UINT8)
        expected = expected.to_bit_depth(BitDepth.UINT8)
        result = cvchan.invert(img)
        self.assertEqual(result, expected)

    def test_invert_errors(self):
        with self.assertRaises(EnforceError):
            cvchan.invert('foo')

    def test_remap_single_channel(self):
        expected = cvdraw.checkerboard(2, 1, (110, 60))[:, :, 'r']
        channels = list('rgb')
        result = cvchan.remap_single_channel(expected, channels)

        self.assertEqual(result.bit_depth, expected.bit_depth)
        self.assertEqual(result.channels, channels)
        for chan in channels:
            self.assert_equal_content(result[:, :, chan], expected)

        channels = [0, 1, 2]
        result = cvchan.remap_single_channel(expected, channels)
        self.assertEqual(result.channels, channels)

    def test_remap_single_channel_errors(self):
        swatch = cvdraw.swatch((10, 10, 3), BasicColor.BLACK)
        expected = 'Image must be an Image with only 1 channel. '
        expected += 'Channels found: 3.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvchan.remap_single_channel(swatch, list('xyz'))

        with self.assertRaises(EnforceError):
            cvchan.remap_single_channel(swatch[:, :, 'r'], 'g')

    def test_remap_one_image(self):
        shape = (10, 10, 3)
        image = cvdraw.swatch(shape, BasicColor.CYAN)
        shape = (10, 10, 1)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        cmap = ChannelMap(dict(v='b', w='w', x='0.r', y='0.g', z='0.b'))

        result = cvchan.remap(image, cmap)

        # attributes
        self.assertEqual(result.width, image.width)
        self.assertEqual(result.height, image.height)
        self.assertEqual(result.bit_depth, image.bit_depth)

        # channels
        self.assertEqual(result.channels, list('vwxyz'))
        self.assert_equal_content(result[:, :, 'v'], black)
        self.assert_equal_content(result[:, :, 'w'], white)
        self.assert_equal_content(result[:, :, 'x'], image[:, :, 'r'])
        self.assert_equal_content(result[:, :, 'y'], image[:, :, 'g'])
        self.assert_equal_content(result[:, :, 'z'], image[:, :, 'b'])

    def test_remap_two_images(self):
        images = [
            cvdraw.swatch(
                (10, 10, 3), BasicColor.CYAN, bit_depth=BitDepth.FLOAT32
            ),
            cvdraw.swatch(
                (10, 10, 4), BasicColor.RED, bit_depth=BitDepth.UINT8
            ),
        ]
        shape = (10, 10, 1)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        cmap = ChannelMap(dict(v='b', w='w', x='1.r', y='0.g', z='0.b'))

        result = cvchan.remap(images, cmap)

        # attributes
        self.assertEqual(result.width, images[0].width)
        self.assertEqual(result.height, images[0].height)
        self.assertEqual(result.bit_depth, images[0].bit_depth)

        # channels
        expected = images[1].to_bit_depth(BitDepth.FLOAT32)[:, :, 'r']

        self.assertEqual(result.channels, list('vwxyz'))
        self.assert_equal_content(result[:, :, 'v'], black)
        self.assert_equal_content(result[:, :, 'w'], white)
        self.assert_equal_content(result[:, :, 'x'], expected)
        self.assert_equal_content(result[:, :, 'y'], images[0][:, :, 'g'])
        self.assert_equal_content(result[:, :, 'z'], images[0][:, :, 'b'])

    def test_remap_multiple_images(self):
        shape = (10, 10, 3)
        images = [
            cvdraw.swatch(shape, BasicColor.RED),
            cvdraw.swatch(shape, BasicColor.GREEN),
            cvdraw.swatch(shape, BasicColor.BLUE),
        ]
        shape = (10, 10, 1)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        cmap = ChannelMap(dict(v='b', w='w', r='0.r', g='1.g', b='2.b'))

        result = cvchan.remap(images, cmap)

        # attributes
        self.assertEqual(result.width, 10)
        self.assertEqual(result.height, 10)
        self.assertEqual(result.bit_depth, images[0].bit_depth)

        # channels
        self.assertEqual(result.channels, list('vwrgb'))
        self.assert_equal_content(result[:, :, 'v'], black)
        self.assert_equal_content(result[:, :, 'w'], white)
        self.assert_equal_content(result[:, :, 'r'], images[0][:, :, 'r'])
        self.assert_equal_content(result[:, :, 'g'], images[1][:, :, 'g'])
        self.assert_equal_content(result[:, :, 'b'], images[2][:, :, 'b'])

    def test_remap_errors(self):
        shape = (10, 10, 3)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        cmap = ChannelMap(dict(r='0.r', g='0.g', b='1.b'))

        # images
        with self.assertRaises(EnforceError):
            cvchan.remap('foo', cmap)

        # heterogenous type
        with self.assertRaises(EnforceError):
            cvchan.remap([black, 'foo'], cmap)

        # irregular width
        white = cvdraw.swatch((9, 10, 3), BasicColor.WHITE)
        with self.assertRaises(EnforceError):
            cvchan.remap([black, white], cmap)

        # irregular height
        white = cvdraw.swatch((10, 9, 3), BasicColor.WHITE)
        with self.assertRaises(EnforceError):
            cvchan.remap([black, white], cmap)

        # ChannelMap
        with self.assertRaises(EnforceError):
            cvchan.remap(black, dict(r='0.r', g='0.g', b='1.b'))
