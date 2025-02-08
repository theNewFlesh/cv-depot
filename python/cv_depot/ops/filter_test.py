import unittest

from lunchbox.enforce import EnforceError
import cv2
import numpy as np

from cv_depot.core.color import BasicColor
from cv_depot.core.image import Image
import cv_depot.ops.draw as cvdraw
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
        with self.assertRaisesRegexp(EnforceError, expected):
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
