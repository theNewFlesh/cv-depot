import unittest

from lunchbox.enforce import EnforceError
import numpy as np

from cv_depot.core.color import BasicColor, Color
from cv_depot.core.image import BitDepth
# ------------------------------------------------------------------------------


class BasicColorTests(unittest.TestCase):
    def test_get_color(self):
        result = BasicColor._get_color('blue', 'string')
        self.assertIs(result, BasicColor.BLUE)

        result = BasicColor._get_color('#00FFFF', 'hexidecimal')
        self.assertIs(result, BasicColor.CYAN)

        result = BasicColor._get_color([0.5], 'one_channel')
        self.assertIs(result, BasicColor.GREY)

        result = BasicColor._get_color([255], 'one_channel_8_bit')
        self.assertIs(result, BasicColor.WHITE)

        result = BasicColor \
            ._get_color([0.0, 1.0, 1.0], 'three_channel')
        self.assertIs(result, BasicColor.CYAN)

        result = BasicColor \
            ._get_color([0, 0, 0], 'three_channel_8_bit')
        self.assertIs(result, BasicColor.BLACK)

        expected = r'is not a legal color\.'
        with self.assertRaisesRegex(ValueError, expected):
            BasicColor._get_color([0, 50, 0], 'three_channel')

        expected = r'is not a legal color\.'
        with self.assertRaisesRegex(ValueError, expected):
            BasicColor._get_color('perriwinkle', 'string')

        expected = r'is not a legal color\.'
        with self.assertRaisesRegex(ValueError, expected):
            BasicColor._get_color('crimson', 'three_channel_8_bit')

    def test_repr(self):
        b = BasicColor.BLACK
        result = repr(b)
        expected = '''
<BasicColor.BLACK>
             string: black
        hexidecimal: #000000
        one_channel: [0.0]
      three_channel: [0.0, 0.0, 0.0]
  one_channel_8_bit: [0]
three_channel_8_bit: [0, 0, 0]'''[1:]
        self.assertEqual(result, expected)

    def test_one_channel_8_bit(self):
        self.assertIsNone(BasicColor.CYAN.one_channel_8_bit)
        self.assertEqual(BasicColor.GREY.one_channel_8_bit, [128])

    def test_three_channel_8_bit(self):
        self.assertEqual(BasicColor.MAGENTA.three_channel_8_bit, [255, 0, 255])
        self.assertEqual(BasicColor.WHITE.three_channel_8_bit, [255, 255, 255])

    def test_string(self):
        self.assertEqual(BasicColor.GREEN.string, 'green')

    def test_from_string(self):
        self.assertIs(BasicColor.from_string('green'), BasicColor.GREEN)
        self.assertIs(BasicColor.from_string('Green'), BasicColor.GREEN)

    def test_from_hexidecimal(self):
        self.assertIs(BasicColor.from_hexidecimal('#00FF00'), BasicColor.GREEN)
        self.assertIs(BasicColor.from_hexidecimal('#00ff00'), BasicColor.GREEN)

    def test_from_list_8_bit(self):
        self.assertIs(
            BasicColor.from_list_8_bit([0, 255, 255]), BasicColor.CYAN
        )
        self.assertIs(
            BasicColor.from_list_8_bit([0.0, 255.0, 255.0]), BasicColor.CYAN
        )

    def test_from_list(self):
        self.assertIs(BasicColor.from_list([1, 0, 1]), BasicColor.MAGENTA)
        self.assertIs(BasicColor.from_list([1.0, 0.0, 1.0]), BasicColor.MAGENTA)

        self.assertIs(BasicColor.from_list([1]), BasicColor.WHITE)
        self.assertIs(BasicColor.from_list([0.5]), BasicColor.GREY)

        expected = r'Invalid color value \[1.0, 0.0, 1.0, 0.0\]\. '
        expected += r'Must be 1 or 3 channels\.'
        with self.assertRaisesRegex(ValueError, expected):
            BasicColor.from_list([1, 0, 1, 0])
# ------------------------------------------------------------------------------


class ColorTests(unittest.TestCase):
    def test_init(self):
        expected = 'Data must be a numpy array.'
        with self.assertRaisesRegex(TypeError, expected):
            Color([1, 2, 3])

        expected = r'Given array has 2 dimensions and a shape of \(2, 3\)\.'
        with self.assertRaisesRegex(AttributeError, expected):
            Color(np.ones((2, 3)))

        expected = r'Given array has 2 dimensions and a shape of \(0, 3\)\.'
        with self.assertRaisesRegex(AttributeError, expected):
            Color(np.ones((0, 3)))

        with self.assertRaises(TypeError):
            Color(np.ones(3, dtype=np.float64))

        # uint8
        x = np.array([0, 128, 255], dtype=np.uint8)
        expected = np.array([0, 0.5, 1], dtype=np.float32).tolist()
        result = Color(x)._data
        self.assertEqual(result.dtype, np.float32)
        result = [round(x, 2) for x in result.tolist()]
        self.assertEqual(result, expected)

        # int8
        x = np.array([-127, 0, 127], dtype=np.int8)
        expected = np.array([0, 0.5, 1], dtype=np.float32).tolist()
        result = Color(x)._data
        self.assertEqual(result.dtype, np.float32)
        result = [round(x, 2) for x in result.tolist()]
        self.assertEqual(result, expected)

        # float32
        expected = np.array([0, 0.5, 1], dtype=np.float32)
        result = Color(expected)._data
        self.assertEqual(result.tolist(), expected.tolist())
        self.assertEqual(result.dtype, np.float32)

        # float16
        expected = np.array([0, 0.5, 1], dtype=np.float16)
        result = Color(expected)._data
        self.assertEqual(result.tolist(), expected.tolist())
        self.assertEqual(result.dtype, np.float32)

    def test_repr(self):
        result = Color.from_list([0, 0, 0]).__repr__()
        expected = '''<Color>
      values: [0. 0. 0.]
   bit_depth: FLOAT32
num_channels: 3
        name: black'''
        self.assertEqual(result, expected)

        result = Color.from_list([0, 1, 1]).__repr__()
        expected = '''<Color>
      values: [0. 1. 1.]
   bit_depth: FLOAT32
num_channels: 3
        name: cyan'''
        self.assertEqual(result, expected)

        result = Color.from_list([0, 0.5, 0]).__repr__()
        expected = '''<Color>
      values: [0.  0.5 0. ]
   bit_depth: FLOAT32
num_channels: 3'''
        self.assertEqual(result, expected)

    def test_from_list(self):
        expected = [0, 0.5, 1]
        result = Color \
            .from_list(expected, bit_depth=BitDepth.FLOAT32) \
            ._data \
            .tolist()
        self.assertEqual(result, expected)

        result = Color \
            .from_list([0, 128, 255], bit_depth=BitDepth.UINT8) \
            ._data \
            .tolist()
        result = [round(x, 2) for x in result]
        expected = [0, 0.5, 1]
        self.assertEqual(result, expected)

        result = Color \
            .from_list([-127, 0, 127], bit_depth=BitDepth.INT8) \
            ._data \
            .tolist()
        result = [round(x, 2) for x in result]
        expected = [0, 0.5, 1]
        self.assertEqual(result, expected)

    def test_from_basic_color(self):
        result = Color.from_basic_color(BasicColor.CYAN)._data.tolist()
        expected = BasicColor.CYAN.three_channel
        self.assertEqual(result, expected)

        result = Color \
            .from_basic_color(BasicColor.CYAN, num_channels=5) \
            ._data \
            .tolist()
        expected = [0.0, 1.0, 1.0, 0.0, 0.0]
        self.assertEqual(result, expected)

        result = Color \
            .from_basic_color(BasicColor.CYAN, num_channels=5, fill_value=0.5) \
            ._data \
            .tolist()
        expected = [0.0, 1.0, 1.0, 0.5, 0.5]
        self.assertEqual(result, expected)

        result = Color \
            .from_basic_color(BasicColor.CYAN, num_channels=2, fill_value=0.5) \
            ._data \
            .tolist()
        expected = [0.0, 1.0]
        self.assertEqual(result, expected)

        result = Color \
            .from_basic_color(BasicColor.BLACK, num_channels=1) \
            ._data \
            .tolist()
        expected = [0.0]
        self.assertEqual(result, expected)

    def test_from_basic_color_errors(self):
        expected = 'num_channels must be greater than or equal to 1. 0 < 1.'
        with self.assertRaisesRegex(EnforceError, expected):
            Color.from_basic_color(BasicColor.CYAN, num_channels=0)

        expected = 'No one channel equivalent found for given color: BasicColor.CYAN.'
        with self.assertRaisesRegex(EnforceError, expected):
            Color.from_basic_color(BasicColor.CYAN, num_channels=1)

    def test_num_channels(self):
        result = Color.from_basic_color(BasicColor.CYAN).num_channels
        self.assertEqual(result, 3)

        result = Color \
            .from_basic_color(BasicColor.BLACK, num_channels=1) \
            .num_channels
        self.assertEqual(result, 1)

        x = np.ones((23), dtype=np.float32)
        result = Color.from_array(x).num_channels
        self.assertEqual(result, 23)

    def test_from_array(self):
        expected = np.array([0, 0.5, 1], dtype=np.float32)
        result = Color.from_array(expected)._data.tolist()
        self.assertEqual(result, expected.tolist())

    def test_from_array_errors(self):
        arr = np.array([0, 0.5, 1], dtype=np.float32)
        with self.assertRaises(EnforceError):
            Color.from_array(arr, num_channels=5.0)

        with self.assertRaises(EnforceError):
            Color.from_array(arr, num_channels=0)

    def test_from_array_num_channels(self):
        expected = [0.0, 0.5, 1.0, 0.25, 0.25]
        arr = np.array(expected[:3], dtype=np.float32)
        result = Color \
            .from_array(arr, num_channels=5, fill_value=0.25) \
            .to_array() \
            .tolist()
        self.assertEqual(result, expected)

    def test_to_array(self):
        color = Color.from_list([0, 0.5, 1])

        result = color.to_array().tolist()
        self.assertEqual(result, [0, 0.5, 1])

        result = color.to_array(bit_depth=BitDepth.UINT8).tolist()
        self.assertEqual(result, [0, 128, 255])

        result = color.to_array(bit_depth=BitDepth.INT8).tolist()
        self.assertEqual(result, [-128, 0, 127])

        bit_depths = [
            BitDepth.UINT8, BitDepth.INT8, BitDepth.FLOAT16, BitDepth.FLOAT32
        ]
        for bit_depth in bit_depths:
            result = color.to_array(bit_depth=bit_depth).dtype
            self.assertEqual(result, bit_depth.dtype)
