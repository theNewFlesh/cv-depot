import unittest

from lunchbox.enforce import EnforceError

from cv_depot.core.channel import ChannelMap
# ------------------------------------------------------------------------------


class ChannelTests(unittest.TestCase):
    def test_setattr(self):
        cmap = ChannelMap()
        cmap['kiwi'] = '0.r'
        self.assertEqual(cmap['kiwi'], '0.r')

        # value is str
        with self.assertRaises(EnforceError):
            cmap['r'] = 0

        # key is str
        with self.assertRaises(EnforceError):
            cmap[1] = '0.r'

        # value matches regex
        expected = 'ChannelMap values must match: b, w, black, white, '
        expected += r'or \[frame\].\[channel\]. Illegal value:'
        with self.assertRaisesRegexp(EnforceError, expected):
            cmap['r'] = 'invalid'

        with self.assertRaisesRegexp(EnforceError, expected):
            cmap['r'] = '-2.r'

        with self.assertRaisesRegexp(EnforceError, expected):
            cmap['r'] = 'r'

    def test_init(self):
        # 1 image
        expected = dict(r='0.r', g='0.g', b='0.b', a='0.a')
        result = list(ChannelMap(expected).items())
        self.assertEqual(result[0], ('r', '0.r'))
        self.assertEqual(result[1], ('g', '0.g'))
        self.assertEqual(result[2], ('b', '0.b'))
        self.assertEqual(result[3], ('a', '0.a'))

        # 2 images
        expected = dict(r='0.r', g='0.g', b='0.b', a='1.a')
        result = list(ChannelMap(expected).items())
        self.assertEqual(result[0], ('r', '0.r'))
        self.assertEqual(result[1], ('g', '0.g'))
        self.assertEqual(result[2], ('b', '0.b'))
        self.assertEqual(result[3], ('a', '1.a'))

        # many images
        expected = dict(r='0.r', g='1.g', b='2.b', a='3.a')
        result = list(ChannelMap(expected).items())
        self.assertEqual(result[0], ('r', '0.r'))
        self.assertEqual(result[1], ('g', '1.g'))
        self.assertEqual(result[2], ('b', '2.b'))
        self.assertEqual(result[3], ('a', '3.a'))

    def test_repr(self):
        result = str(ChannelMap(dict(r='0.r', g='0.g', b='0.b', a='1.a')))
        expected = '''
order target       source
-------------------------
0     r       <--  0.r
1     g       <--  0.g
2     b       <--  0.b
3     a       <--  1.a
'''[1:-1]
        self.assertEqual(result, expected)

    def test_source(self):
        result = ChannelMap(dict(r='0.r', g='0.g', b='0.b', a='1.a')).source
        self.assertEqual(result, ['0.r', '0.g', '0.b', '1.a'])

    def test_target(self):
        result = ChannelMap(dict(r='0.r', g='0.g', b='0.b', a='1.a')).target
        self.assertEqual(result, list('rgba'))
