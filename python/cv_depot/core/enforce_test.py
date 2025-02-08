import unittest

from lunchbox.enforce import EnforceError

import cv_depot.core.enforce as enf
# ------------------------------------------------------------------------------


class EnforceToolsTests(unittest.TestCase):
    def test_enforce_homogenous_type(self):
        enf.enforce_homogenous_type([1, 2, 3, 4])
        enf.enforce_homogenous_type(['foo', 'bar'])
        enf.enforce_homogenous_type(range(10))

        expected = 'Iterable may only contain one type of object. '
        expected += r"Found types: \['int', 'str'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            enf.enforce_homogenous_type([1, 2, 'foo'])

        with self.assertRaisesRegex(EnforceError, expected):
            enf.enforce_homogenous_type(map(lambda x: x, [1, 2, 'foo']))

        expected = 'Foobar may only contain one type of object. '
        expected += r"Found types: \['int', 'str'\]\."
        with self.assertRaisesRegex(EnforceError, expected):
            enf.enforce_homogenous_type([1, 2, 'foo'], name='Foobar')
