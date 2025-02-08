import unittest

from lunchbox.enforce import EnforceError
import lunchbox.tools as lbt
import numpy as np

from cv_depot.core.channel_map import ChannelMap
from cv_depot.core.color import BasicColor, Color
from cv_depot.core.image import BitDepth, Image
import cv_depot.ops.draw as cvdraw
# ------------------------------------------------------------------------------


class DrawTests(unittest.TestCase):
    def get_uv_checker_image(self):
        img = lbt.relative_path(__file__, '../../../resources/uv-checker.png')
        image = Image.read(img)[:100, :100]
        tw = int(image.width / 2)
        th = int(image.height / 2)
        alpha = cvdraw.checkerboard(2, 2, (tw, th))
        cmap = ChannelMap(dict(r='0.r', g='0.g', b='0.b', a='1.r'))
        image = cvdraw.combine([image, alpha], cmap)
        image = image.to_bit_depth(BitDepth.FLOAT32)
        return image

    # SWATCH----------------------------------------------------------------
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

    # CHECKERBOARD----------------------------------------------------------------
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

    # GRID----------------------------------------------------------------------
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
