from pathlib import Path
from tempfile import TemporaryDirectory
import os
import re
import unittest

from lunchbox.enforce import EnforceError
import numpy as np

import cv_depot.core.image as cvimg
import cv_depot.core.video as cvid
# ------------------------------------------------------------------------------


class VideoTests(unittest.TestCase):
    def get_source_path(self, root):
        img = cvimg.Image.from_array(np.zeros((10, 10, 3), dtype=np.uint8))
        target = Path(root, 'test')
        os.makedirs(target)
        for i in range(3):
            tgt = Path(target, f'test_{i:04d}.png')
            img.write(tgt)
        target = Path(target, 'test_{frame}.png').as_posix()
        return target

    def test_get_video_metadata(self):
        with TemporaryDirectory() as root:
            src = self.get_source_path(root)
            tgt = Path(root, 'foo.mp4')
            cvid.write_video(src, tgt)
            result = cvid.get_video_metadata(tgt)
            expected = dict(
                width=10,
                height=10,
                bit_depth='UINT8',
                num_frames=3,
                frame_rate=24,
                codec='h264',
            )
            self.assertEqual(result, expected)

    def test_get_video_metadata_errors(self):
        expected = '/tmp/non-file is not a file or does not exist.'
        with self.assertRaisesRegexp(EnforceError, expected):
            cvid.get_video_metadata('/tmp/non-file')

        with TemporaryDirectory() as root:
            temp = Path(root, 'foo.bar')
            with open(temp, 'w') as f:
                f.write('')

            expected = 'Illegal format: bar. Legal formats: '
            expected += r"\['m4v', 'mov', 'mp4', 'mpg', 'mpeg'\]\."
            with self.assertRaisesRegexp(EnforceError, expected):
                cvid.get_video_metadata(temp)

    def test_write_video(self):
        with TemporaryDirectory() as root:
            src = self.get_source_path(root)
            tgt = Path(root, 'foo.mp4')
            cvid.write_video(src, tgt)
            self.assertTrue(tgt.is_file())
            self.assertGreater(os.stat(tgt).st_size, 0)

    def test_write_video_errors(self):
        with TemporaryDirectory() as root:
            src = self.get_source_path(root)
            bad_src = re.sub('.frame.', 'xxxx', src)
            tgt = '/tmp/foo.mp4'

            # source
            expected = 'Source must be a video file or file pattern with'
            expected += ' {frame} in it. Given value: .*test_xxxx.png.'
            with self.assertRaisesRegexp(EnforceError, expected):
                cvid.write_video(bad_src, tgt)

            # codec
            with self.assertRaises(EnforceError):
                cvid.write_video(src, tgt, codec='h264')

            # codec
            with self.assertRaises(EnforceError):
                cvid.write_video(src, tgt, format_='mp4')

            # framerate
            expected = 'Framerate must be an integer greater than 0. Given '
            expected = 'value: 0.'
            with self.assertRaisesRegexp(EnforceError, expected):
                cvid.write_video(src, tgt, framerate=0)
