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
