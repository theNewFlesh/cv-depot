'''
This moudle is meant as a convenience for programmers who want a clean namespace
to sift through.
'''

import argparse as __argparse
import inspect as __inspect
import re as __re
import types as __types

import cv_depot.ops as __ops  # noqa F401
from cv_depot.core.channel_map import ChannelMap  # noqa F401
from cv_depot.core.color import BasicColor, Color  # noqa F401
from cv_depot.core.enum import BitDepth, ImageFormat, VideoCodec, VideoFormat  # noqa F401
from cv_depot.core.enum import Anchor, Direction  # noqa F401
from cv_depot.core.image import Image, ImageCodec  # noqa F401
from cv_depot.core.viewer import ImageViewer  # noqa F401
# ------------------------------------------------------------------------------


def __create_namespace(module):
    # type: (object) -> __argparse.Namespace
    '''
    Creates a clean namespace from a module.
    Only grabs public functions from the module.

    Args:
        module (object): module

    Returns:
        argparse.Namespace: Clean namespace.
    '''
    params = {}
    for key, val in __inspect.getmembers(module):
        if isinstance(val, __types.FunctionType):
            if __re.search('^[a-z]', key):
                params[key] = val
    return __argparse.Namespace(**params)


ops = __argparse.Namespace(
    channel=__create_namespace(__ops.channel),
    draw=__create_namespace(__ops.draw),
    edit=__create_namespace(__ops.edit),
    filter=__create_namespace(__ops.filter),
)
