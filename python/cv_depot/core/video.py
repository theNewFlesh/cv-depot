from typing import Any, Dict, Union  # noqa F401
from cv_depot.core.types import Filepath  # noqa F401

import os
from pathlib import Path
import re

import ffmpeg
from lunchbox.enforce import Enforce

from cv_depot.core.enum import VideoCodec, VideoFormat
# ------------------------------------------------------------------------------


def get_video_metadata(filepath):
    # type: (Filepath) -> Dict[str, Any]
    '''
    Retrieve video file metadata with ffmpeg.

    Args:
        filepath (str or Path): Path to video file.

    Raises:
        EnforceError: If filepath is not a file or does not exist.
        EnforceError: If filepath format is unsupported.

    Returns:
        dict: Metadata.
    '''
    filepath = Path(filepath)

    msg = f'{filepath} is not a file or does not exist.'
    Enforce(filepath.is_file(), '==', True, message=msg)

    ext = filepath.suffix[1:]
    formats = ['m4v', 'mov', 'mp4', 'mpg', 'mpeg']
    msg = 'Illegal format: {a}. Legal formats: {b}.'
    Enforce(ext.lower(), 'in', formats, message=msg)
    # --------------------------------------------------------------------------

    meta = ffmpeg.probe(filepath.as_posix())['streams'][0]
    lut = {'16': 'FLOAT16', '32': 'FLOAT32', '8': 'UINT8'}
    output = dict(
        width=meta['width'],
        height=meta['height'],
        num_frames=int(meta['nb_frames']),
        frame_rate=int(meta['r_frame_rate'].split('/')[0]),
        bit_depth=lut[meta['bits_per_raw_sample']],
        codec=meta['codec_name'],
    )
    return output


def write_video(
    source,                  # type: Filepath
    target,                  # type: Filepath
    framerate=24,            # type: int
    codec=VideoCodec.H264,   # type: VideoCodec
    format_=VideoFormat.MP4  # type: VideoFormat
):
    # type: (...) -> None
    '''
    Writes given input file or file pattern to given target file.

    Args:
        source (str or Path): Source image filepath or file pattern.
        target (str or Path): Target file.
        framerate (int, optional): Video framerate. Default: 24 fps.
        codec (VideoCodec, optional): Video codec. Default: VideoCodec.H264.
        format_ (VideoFormat, optional): Video container format.
            Default: VideoFormat.MP4

    Raises:
        EnforceError: If source is not a filepath or file pattern.
        EnforceError: framerate is not an integer greater than 0.
        EnforceError: If codec is illegal.
        EnforceError: If format is illegal.
    '''
    source = Path(source).as_posix()
    target = Path(target).as_posix()
    if not Path(source).is_file():
        msg = 'Source must be a video file or file pattern with {a} in it. '
        msg += 'Given value: {b}.'
        Enforce('{frame}', 'in', source, message=msg)
        source = re.sub(r'\{frame\}', '%04d', source)

    msg = 'Framerate must be an integer greater than 0. Given value: {a}.'
    Enforce(framerate, 'instance of', int, message=msg)
    Enforce(framerate, '>', 0, message=msg)

    Enforce(codec, 'instance of', VideoCodec)
    Enforce(format_, 'instance of', VideoFormat)
    # --------------------------------------------------------------------------

    os.makedirs(Path(target).parent, exist_ok=True)
    ffmpeg \
        .input(source, framerate=framerate) \
        .output(target, vcodec=codec.ffmpeg_code, format=format_.extension) \
        .run(overwrite_output=True)
