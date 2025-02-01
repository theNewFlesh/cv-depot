import unittest

from pandas import Series
import numpy as np

import cv_depot.core.tools as cvt
# ------------------------------------------------------------------------------


class ToolsTests(unittest.TestCase):
    def test_get_channels_from_array(self):
        temp = np.zeros((10, 5, 7), dtype=np.float32)
        result = cvt.get_channels_from_array(temp)
        expected = list('rgba') + [4, 5, 6]
        self.assertEqual(result, expected)

        temp = np.zeros((10, 5, 2), dtype=np.float32)
        result = cvt.get_channels_from_array(temp)
        self.assertEqual(result, list('rg'))

        temp = np.zeros((10, 5), dtype=np.float32)
        result = cvt.get_channels_from_array(temp)
        self.assertEqual(result, ['l'])

    def test_apply_minmax(self):
        data = Series([1, 2, 3, 4])
        result = cvt.apply_minmax(data).tolist()
        expected = [0, 1 / 3.0, 2 / 3.0, 1.0]
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])
        self.assertAlmostEqual(result[2], expected[2])
        self.assertAlmostEqual(result[3], expected[3])

        expected = r'Floor must not be greater than ceiling. 10 > 2.'
        with self.assertRaisesRegex(ValueError, expected):
            cvt.apply_minmax(data, floor=10, ceiling=2)

        data = Series([np.nan, 1, 2, 3])
        result = cvt.apply_minmax(data, floor=-1).tolist()
        expected = [0.0, 0.5, 0.75, 1.0]
        self.assertEquals(result, expected)

        data = Series([-np.inf, 1, 2, 3])
        result = cvt.apply_minmax(data, floor=-1).tolist()
        expected = [0.0, 0.5, 0.75, 1.0]
        self.assertEquals(result, expected)

        data = Series([0, 1, 2, np.inf])
        result = cvt.apply_minmax(data, ceiling=4).tolist()
        expected = [0.0, 0.25, 0.5, 1.0]
        self.assertEquals(result, expected)
