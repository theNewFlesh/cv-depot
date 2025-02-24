import unittest

from lunchbox.enforce import EnforceError
import lunchbox.tools as lbt
import numpy as np

from cv_depot.core.channel_map import ChannelMap
from cv_depot.core.color import BasicColor, Color
from cv_depot.core.image import BitDepth, Image
import cv_depot.ops.channel as cvchan
import cv_depot.ops.draw as cvdraw
import cv_depot.ops.edit as cvedit
# ------------------------------------------------------------------------------


class SwatchTests(unittest.TestCase):
    def test_swatch_error(self):
        expected = 'Illegal shape: Each shape dimension must be greater than '
        expected += r'0. Given shape: \(1, 0, 1\).'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.swatch((1, 0, 1), BasicColor.BLACK)

        expected = 'Shape must be an tuple or list of 3 integers. '
        expected += r'Given shape: \[10, 10.0, 3\]. Given type: .*list'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.swatch([10, 10.0, 3], BasicColor.BLACK)

        expected = 'Shape must be an tuple or list of 3 integers. '
        expected += r'Given shape: \(1, 2, 3, 4\). Given type: .*tuple'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.swatch((1, 2, 3, 4), BasicColor.BLACK)

    def test_swatch_basic_color(self):
        result = cvdraw.swatch([10, 5, 3], BasicColor.CYAN)
        self.assertEqual(result.width, 10)
        self.assertEqual(result.height, 5)
        self.assertEqual(result.num_channels, 3)

        expected = np.ones((5, 10, 3), dtype=np.float32)
        expected *= np.array([0, 1, 1], dtype=np.float32)
        expected = Image.from_array(expected)

        self.assertEqual(result, expected)

    def test_swatch_basic_color_extra_channels(self):
        result = cvdraw.swatch([10, 5, 5], BasicColor.CYAN)
        self.assertEqual(result.width, 10)
        self.assertEqual(result.height, 5)
        self.assertEqual(result.num_channels, 5)

        expected = np.ones((5, 10, 3), dtype=np.float32)
        expected *= np.array([0, 1, 1], dtype=np.float32)
        black = np.zeros((5, 10, 2), np.float32)
        expected = np.concatenate([expected, black], axis=2)
        expected = Image.from_array(expected)

        self.assertEqual(result, expected)

    def test_swatch_fill_value(self):
        result = cvdraw.swatch([10, 5, 5], BasicColor.CYAN, fill_value=0.23)
        self.assertEqual(result.width, 10)
        self.assertEqual(result.height, 5)
        self.assertEqual(result.num_channels, 5)

        expected = np.ones((5, 10, 3), dtype=np.float32)
        expected *= np.array([0, 1, 1], dtype=np.float32)
        fill = np.ones((5, 10, 2), np.float32) * 0.23
        expected = np.concatenate([expected, fill], axis=2)
        expected = Image.from_array(expected)

        self.assertEqual(result, expected)

        # color channels less than swatch channels
        red = Color.from_list([1, 0, 0])
        result = cvdraw.swatch([100, 100, 4], red)
        self.assertEqual(result.num_channels, 4)

    def test_swatch_color_bit_depth(self):
        expected = BitDepth.UINT8
        result = cvdraw \
            .swatch([4, 5, 6], BasicColor.RED, bit_depth=expected) \
            .bit_depth
        self.assertEqual(result, expected)

    def test_swatch_color(self):
        color = Color.from_list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = cvdraw.swatch([10, 5, 6], color)

        expected = np.ones((5, 10, 6), dtype=np.float32)
        expected *= color.to_array()
        expected = Image.from_array(expected)

        self.assertEqual(result, expected)


class CheckerboardTests(unittest.TestCase):
    def test_checkerboard(self):
        black = BasicColor.BLACK.three_channel
        white = BasicColor.WHITE.three_channel

        result = cvdraw.checkerboard(5, 10, (3, 4))
        self.assertEqual(result.shape, (15, 40, 3))

        # top row black
        for x in range(0, 15, 6):
            self.assertEqual(result.data[0, x, :].tolist(), black)

        # top row white
        for x in range(3, 15, 6):
            self.assertEqual(result.data[0, x, :].tolist(), white)

        # left column black
        for x in range(0, 10, 8):
            self.assertEqual(result.data[x, 0, :].tolist(), black)

        # left column white
        for x in range(4, 10, 8):
            self.assertEqual(result.data[x, 0, :].tolist(), white)

    def test_checkerboard_errors(self):
        expected = 'Tiles_wide must be greater than 0. 0 !> 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.checkerboard(0, 10, (3, 4))

        expected = 'Tiles_high must be greater than 0. 0 !> 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.checkerboard(10, 0, (3, 4))

        expected = 'Tile width must be greater than 0. 0 !> 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.checkerboard(5, 10, (0, 4))

        expected = 'Tile height must be greater than 0. 0 !> 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.checkerboard(5, 10, (4, 0))


class GridTests(unittest.TestCase):
    def test_grid(self):
        img = cvdraw.swatch((128, 64, 3), BasicColor.BLACK)\
            .to_bit_depth(BitDepth.UINT8)
        result = cvdraw.grid(img, (4, 7), BasicColor.BLUE, 1)
        expected = lbt.relative_path(__file__, '../../../resources/grid.png')
        expected = Image.read(expected).data
        expected = Image.from_array(expected)
        self.assertEqual(result, expected)

        clr = Color.from_basic_color(BasicColor.BLUE)
        result = cvdraw.grid(img, (4, 7), clr, 1)
        self.assertEqual(result, expected)

    def test_grid_errors(self):
        img = cvdraw.swatch((10, 20, 3), BasicColor.BLACK)

        expected = 'Illegal image. str is not an instance of Image.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid('foo', (4, 7), BasicColor.RED, 2)

        expected = r'Illegal shape. Expected \(w, h\)\. Found: \(4, 7, 3\)\.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (4, 7, 3), BasicColor.RED, 2)

        expected = 'Shape height must be greater than or equal to 0. -7 < 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (4, -7), BasicColor.RED, 2)

        expected = 'Shape width must be greater than or equal to 0. -4 < 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (-4, 7), BasicColor.RED, 2)

        expected = r"Color type must be in \['Color', 'BasicColor'\]\. "
        expected += 'Found: tuple.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (4, 7), (255, 0, 0), 2)

        expected = 'Line thickness must be an integer. Found: float.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (4, 7), BasicColor.RED, 0.5)

        expected = '0 is not greater than 0. 0 <= 0.'
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.grid(img, (4, 7), BasicColor.RED, 0)


class HighlightTests(unittest.TestCase):
    def get_uv_checker_image(self):
        img = lbt.relative_path(__file__, '../../../resources/uv-checker.png')
        image = Image.read(img)[:100, :100]
        tw = int(image.width / 2)
        th = int(image.height / 2)
        alpha = cvdraw.checkerboard(2, 2, (tw, th))
        cmap = ChannelMap(dict(r='0.r', g='0.g', b='0.b', a='1.r'))
        image = cvchan.remap([image, alpha], cmap)
        image = image.to_bit_depth(BitDepth.FLOAT32)
        return image

    def get_highlight_image(self):
        red0 = cvdraw.swatch((50, 50, 4), BasicColor.RED, fill_value=0)
        red1 = cvdraw.swatch((50, 50, 4), BasicColor.RED, fill_value=1)
        green0 = cvdraw.swatch((50, 50, 4), BasicColor.GREEN, fill_value=0)
        green1 = cvdraw.swatch((50, 50, 4), BasicColor.GREEN, fill_value=1)
        img0 = cvedit.staple(red0, green0)
        img1 = cvedit.staple(red1, green1)
        image = cvedit.staple(img0, img1, direction='below')

        green = cvdraw.swatch((100, 50, 4), BasicColor.GREEN, fill_value=1)
        expected = cvchan.mix(img1, green)
        expected = cvedit.staple(img0, expected, direction='below')

        i_expected = cvchan.mix(img0, green)
        i_expected = cvedit.staple(i_expected, img1, direction='below')
        return image, expected, i_expected

    def test_highlight(self):
        c0 = cvdraw.checkerboard(2, 2, (50, 50))
        c1 = cvdraw.swatch((100, 100, 3), BasicColor.BLACK)
        c2 = cvchan.invert(c0)
        cmap = ChannelMap(dict(r='0.r', g='1.g', b='1.b', a='2.r'))
        image = cvchan.remap([c0, c1, c2], cmap)

        result = cvdraw.highlight(
            image, mask='a', opacity=1.0, color=BasicColor.RED, inverse=False
        )
        rgb = list('rgb')
        w, h = result.width_and_height
        expected = cvdraw.swatch((w, h, 3), BasicColor.RED)
        self.assertEqual(result[:, :, rgb], expected)

    def test_highlight_identity(self):
        expected = self.get_uv_checker_image()
        result = cvdraw.highlight(expected, mask='a', opacity=0.0)
        self.assertEqual(result, expected)

    def test_highlight_inverse(self):
        image, _, expected = self.get_highlight_image()
        result = cvdraw.highlight(
            image, mask='a', opacity=0.5, color=BasicColor.GREEN, inverse=True
        )
        self.assertEqual(result, expected)

    def test_highlight_opacity(self):
        image, expected, _ = self.get_highlight_image()

        result = cvdraw.highlight(
            image, mask='a', opacity=0.5, color=BasicColor.GREEN, inverse=False
        )

        w, h = image.width_and_height
        w0, w1, w2, w3 = 0, int(0.05 * w), int(0.95 * w), w
        h0, h1, h2, h3 = 0, int(0.05 * h), int(0.95 * h), h
        q0 = (w0, w1), (h0, h1)
        q1 = (w2, w3), (h0, h1)
        q2 = (w0, w1), (h2, h3)
        q3 = (w2, w3), (h2, h3)

        # quadrants with alpha == 1
        for (w0, w1), (h0, h1) in [q2, q3]:
            res = result[w0:w1, h0:h1]
            exp = expected[w0:w1, h0:h1]
            self.assertEqual(res, exp)

            res = result[w0:w1, h0:h1, 'g']
            orig = image[w0:w1, h0:h1, 'g']
            diff = res.compare(orig, content=True)['mean_content_difference']
            self.assertGreaterEqual(diff, 0)

        # quadrants with alpha == 0
        for (w0, w1), (h0, h1) in [q0, q1]:
            res = result[w0:w1, h0:h1]
            exp = image[w0:w1, h0:h1]
            self.assertEqual(res, exp)

    def test_highlight_opacity_one(self):
        color = BasicColor.ORANGE1
        shape = (100, 100, 4)
        image = cvdraw.swatch(shape, BasicColor.BLACK, fill_value=1)
        result = cvdraw.highlight(image, mask='a', opacity=1, color=color)
        expected = cvdraw.swatch(shape, color, fill_value=1)
        self.assertEqual(result, expected)

    def test_highlight_errors(self):
        image = cvdraw.swatch((100, 100, 3), BasicColor.WHITE)
        mask = 'g'
        opacity = 0.4
        color = BasicColor.RED2
        inverse = False

        # image
        with self.assertRaises(EnforceError):
            cvdraw.highlight('foo', mask, opacity, color, inverse)

        with self.assertRaises(EnforceError):
            cvdraw.highlight(image, 'x', opacity, color, inverse)


class OutlineTests(unittest.TestCase):
    def get_outline_image(self, width):
        width += 1
        shape = (50, 100, 4)
        black = cvdraw.swatch(shape, BasicColor.BLACK, fill_value=0)
        white = cvdraw.swatch(shape, BasicColor.WHITE, fill_value=1)
        image = cvedit.staple(black, white)

        green = cvdraw.swatch((width, 100, 4), BasicColor.GREEN, fill_value=1)
        w0 = int(50 - (width / 2))
        w1 = w0 + 1
        shape0 = (w0, 100, 4)
        shape1 = (w1, 100, 4)
        black = cvdraw.swatch(shape0, BasicColor.BLACK, fill_value=0)
        white = cvdraw.swatch(shape1, BasicColor.WHITE, fill_value=0)
        expected = cvedit.staple(black, green)
        expected = cvedit.staple(expected, white)
        return image, expected

    def test_outline(self):
        rgb = list('rgb')
        green = cvdraw.swatch((1, 100, 3), BasicColor.GREEN)
        for width in [10, 20]:
            image, expected = self.get_outline_image(width)
            result = cvdraw.outline(
                image, mask='a', width=width, color=BasicColor.GREEN
            )

            # shape
            self.assertEqual(result.shape, expected.shape)
            self.assertEqual(result.bit_depth, expected.bit_depth)

            # rgb
            r_rgb = result[:, :, rgb]
            e_rgb = expected[:, :, rgb]
            self.assertEqual(r_rgb, e_rgb)

            # alpha
            r_a = result[:, :, 'a']
            e_a = expected[:, :, 'a']
            self.assertEqual(r_a, e_a)

            # all channels
            self.assertEqual(result, expected)

            # w0
            w = (result.width / 2)
            w0 = int(round(w - (width / 2), 0))
            res = result[w0:w0 + 1, :, rgb]
            self.assertEqual(res, green)

            # w1
            w1 = int(round(w + (width / 2), 0)) - 1
            res = result[w1:w1 + 1, :, rgb]
            self.assertEqual(res, green)

    def test_outline_errors(self):
        image, _ = self.get_outline_image(7)
        color = BasicColor.GREEN

        # image
        with self.assertRaises(EnforceError):
            cvdraw.outline('foo', mask='a', width=7, color=color)

        # channels
        expected = 'Mask channel: x not found in image channels: '
        expected += r"\['r', 'g', 'b', 'a'\]."
        with self.assertRaisesRegex(EnforceError, expected):
            cvdraw.outline(image, mask='x', width=7, color=color)

        # width
        with self.assertRaises(EnforceError):
            cvdraw.outline(image, mask='a', width=-1, color=color)


class AnnotateTests(unittest.TestCase):
    def get_outline_image(self, width):
        width += 1
        shape = (50, 100, 4)
        black = cvdraw.swatch(shape, BasicColor.BLACK, fill_value=0)
        white = cvdraw.swatch(shape, BasicColor.WHITE, fill_value=1)
        image = cvedit.staple(black, white)

        green = cvdraw.swatch((width, 100, 4), BasicColor.GREEN, fill_value=1)
        w0 = int(50 - (width / 2))
        w1 = w0 + 1
        shape0 = (w0, 100, 4)
        shape1 = (w1, 100, 4)
        black = cvdraw.swatch(shape0, BasicColor.BLACK, fill_value=0)
        white = cvdraw.swatch(shape1, BasicColor.WHITE, fill_value=0)
        expected = cvedit.staple(black, green)
        expected = cvedit.staple(expected, white)
        return image, expected

    def test_annotate(self):
        image, _ = self.get_outline_image(10)
        params = dict(
            mask='a', opacity=0.4, color=BasicColor.GREEN1, inverse=False
        )
        params1 = dict(mask='a', width=10, color=BasicColor.GREEN1)
        expected = cvdraw.highlight(image, **params)
        expected = cvdraw.outline(expected, **params1)
        rgb = list('rgb')
        expected = expected[:, :, rgb]

        params.update(params1)
        result = cvdraw.annotate(image, **params)
        self.assertEqual(result.channels, expected.channels)
        self.assertEqual(result, expected)

    def test_annotate_keep_mask(self):
        image, _ = self.get_outline_image(10)
        params = dict(
            mask='a', opacity=0.4, color=BasicColor.GREEN1, inverse=False
        )
        params1 = dict(mask='a', width=10, color=BasicColor.GREEN1)
        expected = cvdraw.highlight(image, **params)
        expected = cvdraw.outline(expected, **params1)

        params.update(params1)
        result = cvdraw.annotate(image, keep_mask=True, **params)
        self.assertEqual(result.channels, expected.channels)
        self.assertEqual(result, expected)

    def test_annotate_inverse(self):
        image, _ = self.get_outline_image(10)
        params = dict(
            mask='a', opacity=0.4, color=BasicColor.GREEN1, inverse=True
        )
        params1 = dict(mask='a', width=10, color=BasicColor.GREEN1)
        expected = cvdraw.highlight(image, **params)
        expected = cvdraw.outline(expected, **params1)
        rgb = list('rgb')
        expected = expected[:, :, rgb]

        params.update(params1)
        result = cvdraw.annotate(image, **params)
        self.assertEqual(result.channels, expected.channels)
        self.assertEqual(result, expected)
