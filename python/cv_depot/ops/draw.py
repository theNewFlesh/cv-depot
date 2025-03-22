from typing import Any, List, Tuple, Union  # noqa F401
from cv_depot.core.types import AnyColor  # noqa F401

import copy

from lunchbox.enforce import Enforce
import cv2
import numpy as np

from cv_depot.core.channel_map import ChannelMap
from cv_depot.core.color import BasicColor, Color
from cv_depot.core.image import BitDepth, Image
import cv_depot.ops.channel as cvchan
import cv_depot.ops.draw as cvdraw
import cv_depot.ops.filter as cvfilt
# ------------------------------------------------------------------------------


def swatch(
    shape,                      # type: Union[Tuple[int, int, int], List[int]]
    color,                      # type: AnyColor
    fill_value=0.0,             # type: float
    bit_depth=BitDepth.FLOAT32  # type: BitDepth
):
    # type: (...) -> Image
    '''
    Creates an image of the given shape and color.

    Args:
        shape (tuple[int]): List of 3 ints.
        color (Color): Color of swatch RGB or L channels.
        fill_value (float, optional): Value to fill additional channels with.
            Default: 0.
        bit_depth (BitDepth): Bit depth of swatch. Default: BitDepth.FLOAT32

    Raises:
        EnforceError: If shape is not a tuple of 3 integers.
        EnforceError: If shape has any zero dimensions.

    Returns:
        Image: Color swatch.
    '''
    msg = f'Shape must be an tuple or list of 3 integers. Given shape: {shape}. '
    msg += f'Given type: {type(shape)}.'
    Enforce(shape.__class__.__name__, 'in', ['tuple', 'list'], message=msg)
    Enforce(len(shape), '==', 3, message=msg)
    for item in shape:
        Enforce(item, 'instance of', int, message=msg)

    msg = 'Illegal shape: Each shape dimension must be greater than 0. '
    msg += f'Given shape: {shape}.'
    Enforce(min(shape), '>', 0, message=msg)
    # --------------------------------------------------------------------------

    w, h, c = shape

    if isinstance(color, str):
        color = BasicColor.from_string(color)
    if isinstance(color, BasicColor):
        color = Color.from_basic_color(color)
    color = Color.from_array(
        color.to_array(), num_channels=c, fill_value=fill_value
    )

    output = np.ones((h, w, c), np.float32) * color.to_array()
    output = Image.from_array(output).to_bit_depth(bit_depth)
    return output


def grid(image, shape, color=BasicColor.CYAN.name, thickness=1):
    # type: (Image, Tuple[int, int], AnyColor, int) -> Image
    '''
    Draws a grid on a given image.

    Args:
        image (Image): Image to be drawn on.
        shape (tuple[int]): Width, height tuple.
        color (Color, optional): Color of grid. Default: BasicColor.CYAN.
        thickness (int, optional): Thickness of grid lines in pixels.
            Default: 1.

    Raises:
        EnforceError: If image is not an Image instance.
        EnforceError: If shape is not of length 2.
        EnforceError: If width or height of shape is less than 0:
        EnforceError: If color is not an instance of Color or BasicColor.
        EnforceError: If thickness is not greater than 0.

    Returns:
        Image: Image with grid.
    '''
    msg = 'Illegal image. {a.__class__.__name__} is not an instance of Image.'
    Enforce(image, 'instance of', Image, message=msg)
    msg = f'Illegal shape. Expected (w, h). Found: {shape}.'
    Enforce(len(shape), '==', 2, message=msg)
    msg = 'Shape width must be greater than or equal to 0. {a} < 0.'
    Enforce(shape[0], '>=', 0, message=msg)
    msg = 'Shape height must be greater than or equal to 0. {a} < 0.'
    Enforce(shape[1], '>=', 0, message=msg)
    msg = 'Color type must be in {b}. Found: {a}.'
    Enforce(color.__class__.__name__, 'in', ['Color', 'BasicColor'], message=msg)
    msg = 'Line thickness must be an integer. Found: {a.__class__.__name__}.'
    Enforce(thickness, 'instance of', int, message=msg)
    msg = 'Line thickness must be greater than 0. {a} !> 0.'
    Enforce(thickness, '>', 0)
    # --------------------------------------------------------------------------

    w0, h0 = shape
    w, h = image.width_and_height
    bit_depth = image.bit_depth
    img = image.to_bit_depth(BitDepth.FLOAT32).data.copy()
    if isinstance(color, str):
        color = BasicColor.from_string(color)
    if isinstance(color, BasicColor):
        color = Color.from_basic_color(color)
    clr = color.to_array().tolist()  # type: Any

    # vertical lines
    w_step = int(round(w / w0, 0))
    for x in range(w_step, w, w_step):
        img = cv2.line(img, (x, 0), (x, h), clr, thickness)

    # horizontal lines
    h_step = int(round(h / h0, 0))
    for y in range(h_step, h, h_step):
        img = cv2.line(img, (0, y), (w, y), clr, thickness)

    output = Image.from_array(img).to_bit_depth(bit_depth)
    return output


def checkerboard(tiles_wide, tiles_high, tile_shape=(10, 10)):
    # type: (int, int, Tuple[int, int]) -> Image
    '''
    Draws a checkerboard of given width, height and tile shape.

    Args:
        tiles_wide (int): Number of tiles wide checkerboard will be.
        tiles_high (int): Number of tiles high checkerboard will be.
        tile_shape (tuple[int], optional): Width, height tuple of tile shape.
            Default: (10, 10).

    Raises:
        EnforceError: If tiles_wide is not greater than 0.
        EnforceError: If tiles_high is not greater than 0.
        EnforceError: If tile width is not greater than 0.
        EnforceError: If tile height is not greater than 0.

    Returns:
        Image: Checkerboard image.
    '''
    w, h = tile_shape
    shape = (w, h, 3)

    msg = 'Tiles_wide must be greater than 0. {a} !> 0.'
    Enforce(tiles_wide, '>', 0, message=msg)
    msg = 'Tiles_high must be greater than 0. {a} !> 0.'
    Enforce(tiles_high, '>', 0, message=msg)
    msg = 'Tile width must be greater than 0. {a} !> 0.'
    Enforce(w, '>', 0, message=msg)
    msg = 'Tile height must be greater than 0. {a} !> 0.'
    Enforce(h, '>', 0, message=msg)
    # --------------------------------------------------------------------------

    black = swatch(shape, BasicColor.BLACK).data
    white = swatch(shape, BasicColor.WHITE).data
    even = []
    odd = []
    for i in range(0, tiles_wide):
        if i == 0 or i % 2 == 0:
            even.append(black)
            odd.append(white)
        else:
            even.append(white)
            odd.append(black)

    even = np.concatenate(even, axis=1)
    odd = np.concatenate(odd, axis=1)

    rows = []
    for i in range(0, tiles_high):
        if i == 0 or i % 2 == 0:
            rows.append(even)
        else:
            rows.append(odd)

    output = Image.from_array(np.concatenate(rows, axis=0))
    return output


def highlight(
    image, mask='a', opacity=0.5, color=BasicColor.CYAN2.name, inverse=False
):
    # type: (Image, str, float, AnyColor, bool) -> Image
    '''
    Highlight a masked portion of a given image according to a given channel.

    Args:
        image (Image): Image to be highlighted.
        mask (str, optional): Channel to be used as mask. Default: alpha.
        opacity (float, optional): Opacity of highlight overlayed on image.
            Default: 0.5
        color (Color or BasicColor, optional): Color of highlight.
            Default: BasicColor.CYAN2.
        inverse (bool, optional): Whether to invert the highlight.
            Default: False.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If mask is not an instance of str.
        EnforceError: If mask not found in image channels.
        EnforceError: If opacity is < 0 or > 1.
        EnforceError: If color is not instance of Color.
        EnforceError: If inverse is not a boolean.

    Returns:
        Image: Highlighted image.
    '''
    if isinstance(color, str):
        color = BasicColor.from_string(color)
    if isinstance(color, BasicColor):
        color = Color.from_basic_color(color)

    Enforce(image, 'instance of', Image)
    Enforce(mask, 'instance of', str)
    msg = 'Mask channel: {a} not found in image channels: {b}.'
    Enforce(mask, 'in', image.channels, message=msg)
    Enforce(opacity, '>=', 0)
    Enforce(opacity, '<=', 1)
    Enforce(color, 'instance of', Color)
    Enforce(inverse, 'instance of', bool)
    # --------------------------------------------------------------------------

    img = image.to_bit_depth(BitDepth.FLOAT32)
    channels = copy.deepcopy(image.channels)
    matte = cvchan.remap_single_channel(img[:, :, mask], channels)
    imatte = cvchan.invert(matte)
    if inverse:
        matte, imatte = imatte, matte

    swatch = cvdraw.swatch(image.shape, color, fill_value=1.0)
    data = (image.data * imatte.data) + (swatch.data * matte.data)
    output = Image.from_array(data).to_bit_depth(image.bit_depth)
    output = cvchan \
        .mix(image, output, amount=1 - opacity) \
        .set_channels(image.channels)
    return output


def outline(image, mask='a', width=10, color=BasicColor.CYAN2.name):
    # type (Image, str, int, AnyColor) -> Image
    '''
    Use a given mask to outline a given image.

    Args:
        image (Image): Image with mask channel.
        mask (str, optional): Mask channel. Default: alpha.
        width (int, optional): Outline width. Default: 10.
        color (Color or BasicColor, optional): Color of outline.
            Default: BasicColor.CYAN2

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If channel not found in image channels.
        EnforceError: If width is not >= 0.
        EnforceError: If color is not an instance of Color or BasicColor.

    Returns:
        Image: Image with outline.
    '''
    Enforce(image, 'instance of', Image)
    msg = 'Mask channel: {a} not found in image channels: {b}.'
    Enforce(mask, 'in', image.channels, message=msg)
    Enforce(width, '>=', 0)
    # --------------------------------------------------------------------------

    cmap = ChannelMap({c: f'0.{c}' for c in image.channels})
    cmap[mask] = '1.l'
    w = int(round(width / 2, 0))
    edge = cvfilt.canny_edges(image[:, :, mask], size=w)
    output = cvchan.remap([image, edge], cmap)
    output = highlight(output, mask=mask, opacity=1.0, color=color)
    return output


def annotate(
    image,                        # type: Image
    mask='a',                     # type: str
    opacity=0.5,                  # type: float
    width=10,                     # type: int
    color=BasicColor.CYAN2.name,  # type: AnyColor
    inverse=False,                # type: bool
    keep_mask=False,              # type: bool
):
    # type (...) -> Image
    '''
    Annotate a given image according to a given mask channel.

    Args:
        image (Image): Image with mask channel.
        mask (str, optional): Mask channel. Default: alpha.
        opacity (float, optional): Opacity of annotation. Default: 0.5
        width (int, optional): Outline width. Default: 10.
        color (Color or BasicColor, optional): Color of outline.
            Default: BasicColor.CYAN2
        inverse (bool, optional): Whether to invert the annotation.
            Default: False.
        keep_mask (bool, optional): Whether to keep the mask channel.
            Default: False.

    Returns:
        Image: Image with outline.
    '''
    output = highlight(
        image, mask=mask, opacity=opacity, color=color, inverse=inverse
    )
    output = outline(output, mask=mask, width=width, color=color)
    if not keep_mask:
        channels = copy.deepcopy(image.channels)
        channels.remove(mask)
        output = output[:, :, channels]
    return output
