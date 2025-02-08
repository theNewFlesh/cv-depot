import unittest

from lunchbox.enforce import EnforceError
import numpy as np

from cv_depot.core.color import BasicColor
from cv_depot.core.image import Image
import cv_depot.ops.draw as cvdraw
import cv_depot.ops.edit as cvedit
# ------------------------------------------------------------------------------


class StapleTests(unittest.TestCase):
    def test_staple_direction(self):
        shape = (10, 5, 3)
        a = cvdraw.swatch(shape, BasicColor.RED)
        b = cvdraw.swatch(shape, BasicColor.CYAN)

        expected = 'Illegal direction: top'
        with self.assertRaisesRegex(ValueError, expected):
            cvedit.staple(a, b, direction='top')

    def test_staple_right(self):
        a = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)
        b = cvdraw.swatch((7, 11, 3), BasicColor.GREEN)

        result = cvedit.staple(a, b, 'right')
        expected = Image.from_array(np.concatenate([a.data, b.data], axis=1))
        self.assertEqual(result, expected)

    def test_staple_left(self):
        a = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)
        b = cvdraw.swatch((7, 11, 3), BasicColor.GREEN)

        result = cvedit.staple(a, b, 'left')
        expected = Image.from_array(np.concatenate([b.data, a.data], axis=1))
        self.assertEqual(result, expected)

    def test_staple_above(self):
        a = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)
        b = cvdraw.swatch((7, 11, 3), BasicColor.GREEN)

        result = cvedit.staple(a, b, 'above')
        expected = Image.from_array(np.concatenate([b.data, a.data], axis=0))
        self.assertEqual(result, expected)

    def test_staple_below(self):
        a = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)
        b = cvdraw.swatch((7, 11, 3), BasicColor.GREEN)

        result = cvedit.staple(a, b, 'below')
        expected = Image.from_array(np.concatenate([a.data, b.data], axis=0))
        self.assertEqual(result, expected)

    def test_staple_height(self):
        a = cvdraw.swatch((10, 5, 3), BasicColor.RED)
        b = cvdraw.swatch((5, 10, 3), BasicColor.CYAN)

        expected = 'Image heights must be equal. 5 != 10.'
        with self.assertRaisesRegex(ValueError, expected):
            cvedit.staple(a, b, direction='left')
        with self.assertRaisesRegex(ValueError, expected):
            cvedit.staple(a, b, direction='right')

    def test_staple_width(self):
        a = cvdraw.swatch((10, 5, 3), BasicColor.RED)
        b = cvdraw.swatch((5, 10, 3), BasicColor.CYAN)

        expected = 'Image widths must be equal. 10 != 5.'
        with self.assertRaisesRegex(ValueError, expected):
            cvedit.staple(a, b, direction='above')
        with self.assertRaisesRegex(ValueError, expected):
            cvedit.staple(a, b, direction='below')

    def test_staple_one_channel(self):
        img_a = cvdraw.swatch((7, 11, 1), BasicColor.WHITE)
        img_b = cvdraw.swatch((7, 11, 3), BasicColor.GREEN)
        result = cvedit.staple(img_a, img_b)

        a = img_a.data
        b = img_b.data

        black = np.zeros((11, 7, 2), dtype=np.float32)
        a = np.expand_dims(a, axis=2)
        a = np.concatenate([a, black], axis=2)
        expected = Image.from_array(np.concatenate([a, b], axis=1))
        self.assertEqual(result, expected)

        result = cvedit.staple(img_b, img_a)
        expected = Image.from_array(np.concatenate([b, a], axis=1))
        self.assertEqual(result, expected)

    def test_staple_channel_pad_a_smaller(self):
        a = cvdraw.swatch((7, 11, 5), BasicColor.GREEN)
        b = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)

        result = cvedit.staple(a, b, fill_value=0.27)

        pad = np.ones((11, 7, 2), dtype=np.float32) * 0.27
        b = np.concatenate([b.data, pad], axis=2)
        expected = Image.from_array(np.concatenate([a.data, b], axis=1))

        self.assertEqual(result, expected)

    def test_staple_channel_pad_a_bigger(self):
        a = cvdraw.swatch((7, 11, 3), BasicColor.MAGENTA)
        b = cvdraw.swatch((7, 11, 5), BasicColor.GREEN)

        result = cvedit.staple(a, b, fill_value=0.27)

        pad = np.ones((11, 7, 2), dtype=np.float32) * 0.27
        a = np.concatenate([a.data, pad], axis=2)
        expected = Image.from_array(np.concatenate([a, b.data], axis=1))

        self.assertEqual(result, expected)


class PadTests(unittest.TestCase):
    def test_pad_anchor(self):
        anchors = [
            'top-left',
            'top-center',
            'top-right',
            'center-left',
            'center-center',
            'center-right',
            'bottom-left',
            'bottom-center',
            'bottom-right',
        ]
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        for anchor in anchors:
            cvedit.pad(img, (20, 20, 4), anchor=anchor)

    def test_pad_color(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        temp = cvedit.pad(img, (20, 20, 3), color=BasicColor.RED2)
        temp = np.unique(temp.data.reshape(20 * 20, 3), axis=0).tolist()
        result = []
        for item in temp:
            result.append([round(x, 3) for x in item])
        result = sorted(result)

        expected = [round(x, 3) for x in BasicColor.CYAN2.three_channel]
        self.assertEqual(result[0], expected)

        expected = [round(x, 3) for x in BasicColor.RED2.three_channel]
        self.assertEqual(result[1], expected)

    def test_pad_top_left(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='top-left')

        below = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        right = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, below, 'below')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_top_center(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='top-center')

        below = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        left = cvdraw.swatch((7, 20, 3), BasicColor.BLACK)
        right = cvdraw.swatch((6, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, below, 'below')
        expected = cvedit.staple(expected, left, 'left')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_top_right(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='top-right')

        below = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        left = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, below, 'below')
        expected = cvedit.staple(expected, left, 'left')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_center_left(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='center-left')

        above = cvdraw.swatch((7, 5, 3), BasicColor.BLACK)
        below = cvdraw.swatch((7, 4, 3), BasicColor.BLACK)
        right = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, below, 'below')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_center_center_odd(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='center-center')

        above = cvdraw.swatch((7, 5, 3), BasicColor.BLACK)
        below = cvdraw.swatch((7, 4, 3), BasicColor.BLACK)
        left = cvdraw.swatch((7, 20, 3), BasicColor.BLACK)
        right = cvdraw.swatch((6, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, below, 'below')
        expected = cvedit.staple(expected, left, 'left')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_center_center_even(self):
        img = cvdraw.swatch((8, 12, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='center-center')

        above = cvdraw.swatch((8, 4, 3), BasicColor.BLACK)
        below = cvdraw.swatch((8, 4, 3), BasicColor.BLACK)
        left = cvdraw.swatch((6, 20, 3), BasicColor.BLACK)
        right = cvdraw.swatch((6, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, below, 'below')
        expected = cvedit.staple(expected, left, 'left')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_center_right(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='center-right')

        above = cvdraw.swatch((7, 5, 3), BasicColor.BLACK)
        below = cvdraw.swatch((7, 4, 3), BasicColor.BLACK)
        left = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, below, 'below')
        expected = cvedit.staple(expected, left, 'left')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_bottom_left(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='bottom-left')

        above = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        right = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_bottom_center(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='bottom-center')

        above = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        left = cvdraw.swatch((7, 20, 3), BasicColor.BLACK)
        right = cvdraw.swatch((6, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, left, 'left')
        expected = cvedit.staple(expected, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_bottom_right(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 20, 3), anchor='bottom-right')

        above = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)
        left = cvdraw.swatch((13, 20, 3), BasicColor.BLACK)
        expected = cvedit.staple(img, above, 'above')
        expected = cvedit.staple(expected, left, 'left')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_same_height(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (20, 11, 3), anchor='top-left')
        right = cvdraw.swatch((13, 11, 3), BasicColor.BLACK)

        expected = cvedit.staple(img, right, 'right')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_same_width(self):
        img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
        result = cvedit.pad(img, (7, 20, 3), anchor='bottom-left')
        above = cvdraw.swatch((7, 9, 3), BasicColor.BLACK)

        expected = cvedit.staple(img, above, 'above')

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result, expected)

    def test_pad_shape_len_error(self):
        expected = r'Shape must be of length 3\. Given shape: \(20, 20\)\.'
        with self.assertRaisesRegex(ValueError, expected):
            img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
            cvedit.pad(img, (20, 20))

    def test_pad_shape_height_error(self):
        expected = 'Output shape must be greater than or equal to input shape '
        expected += r'in each dimension. \(20, 6, 3\) !>= \(7, 11, 3\)\.'
        with self.assertRaisesRegex(ValueError, expected):
            img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
            cvedit.pad(img, (20, 6, 3))

    def test_pad_shape_width_error(self):
        expected = 'Output shape must be greater than or equal to input shape '
        expected += r'in each dimension. \(3, 20, 3\) !>= \(7, 11, 3\)\.'
        with self.assertRaisesRegex(ValueError, expected):
            img = cvdraw.swatch((7, 11, 3), BasicColor.CYAN2)
            cvedit.pad(img, (3, 20, 3))


class CutTests(unittest.TestCase):
    def test_cut_vertical(self):
        a = cvdraw.swatch((5, 10, 3), BasicColor.BLACK)
        b = cvdraw.swatch((5, 10, 3), BasicColor.WHITE)
        img = cvedit.staple(a, b, direction='right')
        ra, rb = cvedit.cut(img, 5, axis='vertical')

        self.assertEqual(ra.shape, a.shape)
        self.assertEqual(rb.shape, b.shape)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    def test_cut_horizontal(self):
        a = cvdraw.swatch((10, 5, 3), BasicColor.BLACK)
        b = cvdraw.swatch((10, 5, 3), BasicColor.WHITE)
        img = cvedit.staple(a, b, direction='below')
        ra, rb = cvedit.cut(img, 5, axis='horizontal')

        self.assertEqual(ra.shape, a.shape)
        self.assertEqual(rb.shape, b.shape)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    def test_cut_vertical_three_images(self):
        a = cvdraw.swatch((5, 10, 3), BasicColor.RED)
        b = cvdraw.swatch((5, 10, 3), BasicColor.WHITE)
        c = cvdraw.swatch((5, 10, 3), BasicColor.BLUE)
        img = cvedit.staple(a, b, direction='right')
        img = cvedit.staple(img, c, direction='right')

        ra, rb, rc = cvedit.cut(img, [5, 10], axis='vertical')

        self.assertEqual(ra.shape, a.shape)
        self.assertEqual(rb.shape, b.shape)
        self.assertEqual(rc.shape, c.shape)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)
        self.assertEqual(rc, c)

    def test_cut_horizontal_three_images(self):
        a = cvdraw.swatch((10, 5, 3), BasicColor.RED)
        b = cvdraw.swatch((10, 5, 3), BasicColor.WHITE)
        c = cvdraw.swatch((10, 5, 3), BasicColor.BLUE)
        img = cvedit.staple(a, b, direction='below')
        img = cvedit.staple(img, c, direction='below')

        ra, rb, rc = cvedit.cut(img, [5, 10], axis='horizontal')

        self.assertEqual(ra.shape, a.shape)
        self.assertEqual(rb.shape, b.shape)
        self.assertEqual(rc.shape, c.shape)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)
        self.assertEqual(rc, c)

    def test_cut_out_of_bounds(self):
        img = cvdraw.swatch((10, 20, 3), BasicColor.BLACK)

        expected = 'Index out of bounds. 15 > 10.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, 15, axis='vertical')

        expected = 'Index out of bounds. 11 > 10.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, 11, axis='vertical')

        expected = 'Index out of bounds. -1 < 0.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, -1, axis='vertical')

        expected = 'Index out of bounds. 25 > 20.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, 25, axis='horizontal')

        expected = 'Index out of bounds. 21 > 20.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, 21, axis='horizontal')

        expected = 'Index out of bounds. -1 < 0.'
        with self.assertRaisesRegex(IndexError, expected):
            cvedit.cut(img, -1, axis='horizontal')

    def test_cut_bad_axis(self):
        img = cvdraw.swatch((10, 10, 3), BasicColor.BLACK)
        expected = 'Illegal axis: foo. Legal axes include: '
        expected += r"\['vertical', 'horizontal'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            cvedit.cut(img, 15, axis='foo')


class ChopTests(unittest.TestCase):
    def get_chop_data(self):
        shape = (50, 100, 3)
        white = cvdraw.swatch(shape, BasicColor.WHITE)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        row0 = cvedit.staple(black, white, direction='right')
        row1 = cvedit.staple(black, black, direction='right')
        image = cvedit.staple(row0, row1, direction='above')
        lut = {
            (0, 0): black,
            (0, 1): black,
            (1, 0): white,
            (1, 1): black,
        }
        return image, lut

    def get_color_chop_data(self):
        shape = (50, 100, 3)
        blue = cvdraw.swatch(shape, BasicColor.BLUE)
        red = cvdraw.swatch(shape, BasicColor.RED)
        black = cvdraw.swatch(shape, BasicColor.BLACK)
        row0 = cvedit.staple(black, blue, direction='right')
        row1 = cvedit.staple(red, black, direction='right')
        image = cvedit.staple(row0, row1, direction='above')
        lut = {
            (0, 0): black,
            (0, 1): red,
            (1, 0): blue,
            (1, 1): black,
        }
        return image, lut

    def test_chop_channel(self):
        image, lut = self.get_color_chop_data()

        # blue
        result = cvedit.chop(image, channel='b', mode='vertical-horizontal')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (1, 0), (1, 1)])
        )

        e00 = cvedit.staple(lut[(0, 0)], lut[(0, 1)], direction='above')
        e10 = lut[(1, 0)]
        e11 = lut[(1, 1)]
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(1, 0)], e10)
        self.assertEqual(result[(1, 1)], e11)

        # red
        result = cvedit.chop(image, channel='r', mode='vertical-horizontal')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (0, 1), (1, 0)])
        )

        e00 = lut[(0, 0)]
        e01 = lut[(0, 1)]
        e10 = cvedit.staple(lut[(1, 0)], lut[(1, 1)], direction='above')
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(0, 1)], e01)
        self.assertEqual(result[(1, 0)], e10)

    def test_chop_array(self):
        image, _ = self.get_chop_data()
        cvedit.chop(image.data, channel='b', mode='vertical')

    def test_chop_errors(self):
        with self.assertRaises(EnforceError):
            cvedit.chop('foo')

        image, _ = self.get_chop_data()
        expected = 'pizza is not a valid channel. Channels include: '
        expected += r"\['r', 'g', 'b'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            cvedit.chop(image, channel='pizza')

        expected = 'width is not a legal mode. Legal modes include: .*'
        with self.assertRaisesRegex(EnforceError, expected):
            cvedit.chop(image, channel='b', mode='width')

    def test_chop_mode_vertical(self):
        image, lut = self.get_chop_data()
        result = cvedit.chop(image, channel='b', mode='vertical')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (1, 0)])
        )

        e00 = cvedit.staple(lut[(0, 0)], lut[(0, 1)], direction='above')
        e10 = cvedit.staple(lut[(1, 0)], lut[(1, 1)], direction='above')
        self.assertEqual(result[(0, 0)].shape, e00.shape)
        self.assertEqual(result[(1, 0)].shape, e10.shape)
        self.assertEqual(result[(0, 0)].bit_depth, e00.bit_depth)
        self.assertEqual(result[(1, 0)].bit_depth, e10.bit_depth)
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(1, 0)], e10)

    def test_chop_mode_horizontal(self):
        image, lut = self.get_chop_data()
        result = cvedit.chop(image, channel='b', mode='horizontal')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (0, 1)])
        )

        e00 = cvedit.staple(lut[(0, 0)], lut[(1, 0)], direction='right')
        e01 = cvedit.staple(lut[(0, 1)], lut[(1, 1)], direction='right')
        self.assertEqual(result[(0, 0)].shape, e00.shape)
        self.assertEqual(result[(0, 1)].shape, e01.shape)
        self.assertEqual(result[(0, 0)].bit_depth, e00.bit_depth)
        self.assertEqual(result[(0, 1)].bit_depth, e01.bit_depth)
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(0, 1)], e01)

    def test_chop_mode_vertical_horizontal(self):
        image, lut = self.get_chop_data()
        result = cvedit.chop(image, channel='b', mode='vertical-horizontal')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (1, 0), (1, 1)])
        )

        e00 = cvedit.staple(lut[(0, 0)], lut[(0, 1)], direction='above')
        e10 = lut[(1, 0)]
        e11 = lut[(1, 1)]
        self.assertEqual(result[(0, 0)].shape, e00.shape)
        self.assertEqual(result[(1, 0)].shape, e10.shape)
        self.assertEqual(result[(1, 1)].shape, e11.shape)
        self.assertEqual(result[(0, 0)].bit_depth, e00.bit_depth)
        self.assertEqual(result[(1, 0)].bit_depth, e10.bit_depth)
        self.assertEqual(result[(1, 1)].bit_depth, e11.bit_depth)
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(1, 0)], e10)
        self.assertEqual(result[(1, 1)], e11)

    def test_chop_mode_horizontal_vertical(self):
        image, lut = self.get_chop_data()
        result = cvedit.chop(image, channel='b', mode='horizontal-vertical')

        self.assertEqual(
            set(result.keys()),
            set([(0, 0), (1, 0), (0, 1)])
        )

        e00 = lut[(0, 0)]
        e10 = lut[(1, 0)]
        e01 = cvedit.staple(lut[(0, 1)], lut[(1, 1)], direction='right')
        self.assertEqual(result[(0, 0)].shape, e00.shape)
        self.assertEqual(result[(1, 0)].shape, e10.shape)
        self.assertEqual(result[(0, 1)].shape, e01.shape)
        self.assertEqual(result[(0, 0)].bit_depth, e00.bit_depth)
        self.assertEqual(result[(1, 0)].bit_depth, e10.bit_depth)
        self.assertEqual(result[(0, 1)].bit_depth, e01.bit_depth)
        self.assertEqual(result[(0, 0)], e00)
        self.assertEqual(result[(1, 0)], e10)
        self.assertEqual(result[(0, 1)], e01)
