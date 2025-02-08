from typing import Union  # noqa F401

import re

from lunchbox.enforce import Enforce
import cv2
import numpy as np

from cv_depot.core.channel_map import ChannelMap
from cv_depot.core.color import BasicColor
from cv_depot.core.enum import BitDepth
from cv_depot.core.image import Image
import cv_depot.core.image as cvimg
import cv_depot.ops.draw as cvdraw
# ------------------------------------------------------------------------------


def has_super_brights(image):
    # type: (Image) -> bool
    '''
    Determines if given image has values above 1.0.

    Args:
        image (Image): Image instance.

    Raises:
        EnforceError: If image is not an Image instance.

    Returns:
        bool: Presence of super brights.
    '''
    return cvimg._has_super_brights(image)


def has_super_darks(image):
    # type: (Image) -> bool
    '''
    Determines if given image has values below 0.0.

    Args:
        image (Image): Image instance.

    Raises:
        EnforceError: If image is not an Image instance.

    Returns:
        bool: Presence of super darks.
    '''
    return cvimg._has_super_darks(image)


def to_hsv(image):
    # type: (Image) -> Image
    '''
    Convert image to hue, saturation, value colorspace.

    Args:
        Image: Image to be converted.

    Raises:
        AttributeError: If given image does not have RGB channels.

    Returns:
        Image: Image converted to HSV.
    '''
    rgb = list('rgb')
    channels = set(image.channels).intersection(rgb)
    if channels != set(rgb):
        msg = 'Image does not contain RGB channels. '
        msg += f'Channels found: {image.channels}.'
        raise AttributeError(msg)

    img_ = image.to_bit_depth(BitDepth.FLOAT32)
    img = img_[:, :, rgb].data
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 0] = np.divide(img[:, :, 0], 360)
    output = Image\
        .from_array(img)\
        .set_channels(list('hsv'))\
        .to_bit_depth(image.bit_depth)
    return output


def to_rgb(image):
    # type: (Image) -> Image
    '''
    Convert image from HSV to RGB.

    Args:
        Image: Image to be converted.

    Raises:
        AttributeError: If given image does not have RGB channels.

    Returns:
        Image: Image converted to RGB.
    '''
    hsv = list('hsv')
    channels = set(image.channels).intersection(hsv)
    if channels != set(hsv):
        msg = 'Image does not contain HSV channels. '
        msg += f'Channels found: {image.channels}.'
        raise AttributeError(msg)

    img_ = image.to_bit_depth(BitDepth.FLOAT32)
    img = img_[:, :, hsv].data
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img[:, :, 0] = np.divide(img[:, :, 0], 360)
    output = Image\
        .from_array(img)\
        .set_channels(list('rgb'))\
        .to_bit_depth(image.bit_depth)
    return output


def invert(image):
    # type: (Image) -> Image
    '''
    Inverts the values of the given image.
    Black becomes white, white becomes black.

    Args:
        image (Image): Image to be inverted.

    Raises:
        EnforeError: If image is not an instance of Image.

    Returns:
        image: Image
    '''
    Enforce(image, 'instance of', Image)
    # --------------------------------------------------------------------------

    bit_depth = image.bit_depth
    data = image.to_bit_depth(BitDepth.FLOAT32).data
    data = data * -1 + 1
    output = Image.from_array(data).to_bit_depth(bit_depth)
    return output


def mix(a, b, amount=0.5):
    # type: (Image, Image, float) -> Image
    '''
    Mix images A and B by a given amount.
    An amount of 1.0 means 100% of image A.
    An amount of 0.0 means 100% of image B.

    Args:
        a (Image): Image A.
        b (Image): Image B.
        amount (float, optional): Amount of image A. Default: 0.5

    Raises:
        EnforceError: If a is not an Image instance.
        EnforceError: If b is not an Image instance.
        EnforceError: If amount is not between 0 and 1.

    Returns:
        Image: Mixture of A and B.
    '''
    Enforce(a, 'instance of', Image)
    Enforce(b, 'instance of', Image)
    Enforce(amount, '<=', 1)
    Enforce(amount, '>=', 0)
    # --------------------------------------------------------------------------

    amount = float(amount)
    bit_depth = a.bit_depth
    x = a.to_bit_depth(BitDepth.FLOAT32).data
    y = b.to_bit_depth(BitDepth.FLOAT32).data
    img = x * amount + y * (1 - amount)
    output = Image.from_array(img).to_bit_depth(bit_depth)
    return output


def remap_single_channel(image, channels):
    # type: (Image, list) -> Image
    '''
    Maps an image with a single channel to an image of a given number of
    channels.

    Args:
        image (Image): Image to be mapped.
        channels (list): List of channel names to map image to.

    Raises:
        EnforceError: If image is not an Image with only one channel.
        EnforceError: If channels is not a list.

    Returns:
        Image: Image with given channels.
    '''
    Enforce(image, 'instance of', Image)
    msg = 'Image must be an Image with only 1 channel. '
    msg += 'Channels found: {a}.'
    Enforce(image.num_channels, '==', 1, message=msg)
    Enforce(channels, 'instance of', list)
    # --------------------------------------------------------------------------

    data = np.squeeze(image.data)[..., np.newaxis]
    output = np.concatenate([data] * len(channels), axis=2)
    output = Image.from_array(output).set_channels(channels)
    return output


def remap(images, channel_map):
    # type: (Union[Image, list[Image]], ChannelMap) -> Image
    '''
    Maps many images into a single image according to a given channel
    map.

    Args:
        images (Image, list[Image]): Images.
        channel_map (ChannelMap): Mapping of image channels into output image.

    Raises:
        EnforceError: If images is not an instance of Image or list of Images.
        EnforceError: If images are not of all the same width and height.
        EnforceError: If channel_map is not an instance of ChannelMap.

    Returns:
        Image: Combined image.
    '''
    if isinstance(images, Image):
        images = [images]

    msg = 'Images must be an Image or list of Images of uniform width and height.'
    msg += f' Given type: {images.__class__.__name__}'
    [Enforce(x, 'instance of', Image) for x in images]

    shapes = {x.width_and_height for x in images}
    Enforce(len(shapes), '==', 1, message=msg)

    Enforce(channel_map, 'instance of', ChannelMap)
    # --------------------------------------------------------------------

    w, h = images[0].width_and_height
    bit_depth = images[0].bit_depth
    images = [x.to_bit_depth(bit_depth) for x in images]

    shape = (w, h, 1)
    black = cvdraw.swatch(shape, BasicColor.BLACK, bit_depth=bit_depth)
    white = cvdraw.swatch(shape, BasicColor.WHITE, bit_depth=bit_depth)

    channels = []
    for chan in channel_map.source:
        chan_l = chan.lower()
        img = None

        if chan_l in ['b', 'black']:
            img = black
        elif chan_l in ['w', 'white']:
            img = white
        else:
            frame, tgt_chan = re.split(r'\.', chan, maxsplit=1)
            img = images[int(frame)][:, :, tgt_chan]

        array = np.squeeze(img.data)[..., np.newaxis]
        channels.append(array)

    img = np.concatenate(channels, axis=2).astype(bit_depth.dtype)
    output = Image.from_array(img).set_channels(channel_map.target)  # type: ignore
    return output
