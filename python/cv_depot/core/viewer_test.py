import unittest

import numpy as np

from cv_depot.core.image import Image
from cv_depot.core.viewer import ImageViewer
# ------------------------------------------------------------------------------


class ImageViewerTests(unittest.TestCase):
    def get_image(self):
        array = np.zeros((10, 5, 6), dtype=np.uint8)
        image = Image.from_array(array)
        image = image.set_channels([
            'diffuse.x', 'diffuse.y', 'diffuse.z',
            'spec.x', 'spec.y', 'spec.z'
        ])
        return image

    def test_init(self):
        image = self.get_image()
        result = ImageViewer(image, 50, 0.5)
        self.assertIs(result._image, image)
        self.assertEqual(result.layer, 'diffuse')
        self.assertEqual(result.channel, 'all')
        self.assertEqual(result.size, 50)
        self.assertEqual(result.gamma, 0.5)
        self.assertTrue(hasattr(result, 'layer_selector'))
        self.assertTrue(hasattr(result, 'channel_selector'))
        self.assertTrue(hasattr(result, 'viewer'))
        self.assertTrue(hasattr(result, 'info'))
        self.assertTrue(hasattr(result, '_widgets'))

    def test_get_layer_options(self):
        result = ImageViewer(self.get_image())._get_layer_options()
        self.assertEqual(result, ['diffuse', 'spec'])

    def test_get_channel_options(self):
        result = ImageViewer(self.get_image())._get_channel_options()
        self.assertEqual(result, ['all', 'diffuse.x', 'diffuse.y', 'diffuse.z'])

    def test_get_info(self):
        result = ImageViewer(self.get_image())._get_info()
        self.assertIn('UINT8', result)

    def test_get_png(self):
        result = ImageViewer(self.get_image())._get_png()
        self.assertIsInstance(result, bytes)

    def test_handle_layer_event(self):
        viewer = ImageViewer(self.get_image())
        event = dict(type='change', new='spec')
        viewer._handle_layer_event(event)

        self.assertEqual(viewer.layer, 'spec')
        self.assertEqual(viewer.channel, 'all')

        self.assertEqual(viewer.layer_selector.value, 'spec')
        self.assertEqual(viewer.channel_selector.label, 'all')
        self.assertEqual(viewer.channel_selector.value, 'all')
        expected = ('all', 'spec.x', 'spec.y', 'spec.z')
        self.assertEqual(viewer.channel_selector.options, expected)

    def test_handle_channel_event(self):
        viewer = ImageViewer(self.get_image())
        event = dict(type='change', new='diffuse.x')
        viewer._handle_channel_event(event)

        self.assertEqual(viewer.channel_selector.value, 'diffuse.x')

    def test_handle_resize_event(self):
        viewer = ImageViewer(self.get_image())
        event = dict(type='change', new=42)
        viewer._handle_resize_event(event)

        self.assertEqual(viewer.size_slider.value, 42)
        self.assertEqual(viewer.viewer.width, '42%')

    def test_handle_gamma_event(self):
        viewer = ImageViewer(self.get_image())
        event = dict(type='change', new=1.2)
        viewer._handle_gamma_event(event)

        self.assertEqual(viewer.gamma_slider.value, 1.2)
