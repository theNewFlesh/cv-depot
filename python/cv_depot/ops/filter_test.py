import unittest

from lunchbox.enforce import EnforceError
import numpy as np

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
