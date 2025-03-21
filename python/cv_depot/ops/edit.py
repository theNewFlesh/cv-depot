from typing import Any  # noqa F401
from cv_depot.core.types import AnyAnchor, AnyColor, OptArray  # noqa F401

import math

from lunchbox.enforce import Enforce
import cv2
import numpy as np

from cv_depot.core.color import BasicColor
from cv_depot.core.enum import Anchor
from cv_depot.core.image import BitDepth, Image
import cv_depot.core.enforce as enf
import cv_depot.ops.draw as cvdraw
# ------------------------------------------------------------------------------


def reformat(image, width, height):
    # type: (Image, float, float) -> Image
    '''
    Reformat given image by given width and height factors.

    Args:
        image (Image): Image instance.
        width (float): Factor to scale image width by.
        height (float): Factor to scale image height by.

    Raises:
        EnforceError: If image is not an Image instance.
        ValueError: If width or height of reformatted image is less than 1 pixel.

    Returns:
        Image: Reformatted image.
    '''
    Enforce(image, 'instance of', Image)
    # --------------------------------------------------------------------------

    source_bit_depth = image.bit_depth
    image = image.to_bit_depth(BitDepth.FLOAT32)

    x = round(image.width * width)
    y = round(image.height * height)
    if x < 1 or y < 1:
        msg = 'Invalid scale factors. Width and height must be at least 1 pixel'
        msg += f'. Format shape in pixels: ({width}, {height}).'
        raise ValueError(msg)

    tmp = cv2.resize(image.data, (x, y))
    output = Image.from_array(tmp).to_bit_depth(source_bit_depth)
    return output


def crop(image, width_mult, height_mult, width_offset=0, height_offset=0):
    # type: (Image, float, float, int, int) -> Image
    '''
    Crop a given image according to a width and height multipliers and offsets.

    Args:
        image (Image): Image to be cropped.
        width_mult (float): Width multiplier.
        height_mult (float): Height multiplier.
        width_offset (float): Width offset.
        height_offset (float): Height offset.

    Raises:
        EnforeError: If image is not an instance of Image.
        EnforeError: If width_mult is not > 0 and <= 1.
        EnforeError: If height_mult is not > 0 and <= 1.
        EnforceError: If crop dimensions are 0 in width or height.
        EnforceError: If crop width bounds are outside image dimensions.
        EnforceError: If crop height bounds are outside image dimensions.

    Returns:
        Image: Cropped image.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(width_mult, '>', 0)
    Enforce(width_mult, '<=', 1)
    Enforce(height_mult, '>', 0)
    Enforce(height_mult, '<=', 1)
    # --------------------------------------------------------------------------

    height_offset *= -1

    w, h = image.width_and_height
    cw = w * 0.5
    ch = h * 0.5
    wb = cw * width_mult
    hb = ch * height_mult

    w0 = int((cw - wb) + width_offset)
    w1 = int(cw + wb + width_offset)
    h0 = int((ch - hb) + height_offset)
    h1 = int(ch + hb + height_offset)

    cw = int(cw)
    ch = int(ch)

    # ensure crop bounds are within image dimensions
    dw = int(w * width_mult)
    dh = int(h * height_mult)
    msg = 'Crop width and height must be greater than 0. '
    msg += f'Crop dimensions: {dw}, {dh}'
    Enforce(w * width_mult, '>=', 1, message=msg)
    Enforce(h * height_mult, '>=', 1, message=msg)

    msg = f'Crop width bounds: ({w0 - cw}, {w1 - cw}) outside of '
    msg += f'image width bounds: ({-cw}, {cw})'
    Enforce(w0, '>=', 0, message=msg)
    Enforce(w1, '>=', 0, message=msg)
    Enforce(w0, '<=', w, message=msg)
    Enforce(w1, '<=', w, message=msg)

    msg = f'Crop height bounds: ({h0 - ch}, {h1 - ch}) outside of '
    msg += f'image height bounds: ({-ch}, {ch})'
    Enforce(h0, '>=', 0, message=msg)
    Enforce(h1, '>=', 0, message=msg)
    Enforce(h0, '<=', h, message=msg)
    Enforce(h1, '<=', h, message=msg)

    return image[w0:w1, h0:h1]


def pad(image, shape, anchor=Anchor.TOP_LEFT, color=BasicColor.BLACK.name):
    # type: (Image, tuple[int, int, int], AnyAnchor, AnyColor) -> Image
    '''
    Pads a given image into a new image of a given shape.

    The anchor argument determines which corner of the given image will be
    anchored to the padded images respective corner. For instance, an anchor of
    'top-left' will anchor the top-left corner of the given image to the
    top-left corner of the padded image. 'center-left' will vertically center
    the image and horizontally anchor to the left of the image. 'center-center'
    will vertically and horizontally center the image.

    Args:
        image (Image): Image to be padded.
        shape (tuple[int]): Shape (width, height, channels) of padded image.
        anchor (Anchor or str, optional): Where the given image will be placed within the
            new image. Default: top-left.
        color (Color or BasicColor, optional): Padding color.
            Default: BasicColor.BLACK

    Returns:
        Image: Padded image.
    '''
    if len(shape) != 3:
        msg = f'Shape must be of length 3. Given shape: {shape}.'
        raise ValueError(msg)

    w, h, c = image.shape
    dw = shape[0] - w
    dh = shape[1] - h
    c = max(c, shape[2])

    if dw < 0 or dh < 0:
        msg = 'Output shape must be greater than or equal to input shape in each'
        msg += f' dimension. {shape} !>= {image.shape}.'
        raise ValueError(msg)
    # --------------------------------------------------------------------------

    # resolve anchor
    if isinstance(anchor, str):
        dir_h, dir_w = Anchor.from_string(anchor).value
    else:
        dir_h, dir_w = anchor.value

    lut = dict(
        center='center', top='below', bottom='above', left='right', right='left'
    )
    dir_h = lut[dir_h]
    dir_w = lut[dir_w]
    # --------------------------------------------------------------------------

    bit_depth = image.bit_depth
    output = image.to_bit_depth(BitDepth.FLOAT32)
    if dh > 0:
        if dir_h != 'center':
            pad_h = cvdraw.swatch((output.width, dh, c), color)
            output = staple(output, pad_h, direction=dir_h)

        else:
            ph0 = (output.width, math.ceil(dh / 2), c)
            ph1 = (output.width, math.floor(dh / 2), c)

            pad_h0 = cvdraw.swatch(ph0, color)
            output = staple(output, pad_h0, direction='above')

            if ph1[1] > 0:
                pad_h1 = cvdraw.swatch(ph1, color)
                output = staple(output, pad_h1, direction='below')

    if dw > 0:
        if dir_w != 'center':
            pad_w = cvdraw.swatch((dw, output.height, c), color)
            output = staple(output, pad_w, direction=dir_w)

        else:
            pw0 = (math.ceil(dw / 2), output.height, c)
            pw1 = (math.floor(dw / 2), output.height, c)

            pad_w0 = cvdraw.swatch(pw0, color)
            output = staple(output, pad_w0, direction='left')

            if pw1[0] > 0:
                pad_w1 = cvdraw.swatch(pw1, color)
                output = staple(output, pad_w1, direction='right')

    output = output.to_bit_depth(bit_depth)
    return output


def staple(image_a, image_b, direction='right', fill_value=0.0):
    # type: (Image, Image, str, float) -> Image
    '''
    Joins two images along a given direction.

    .. image:: images/staple.png

    Images must be the same height if stapling along left/right axis.
    Images must be the same width if stapling along above/below axis.

    Args:
        image_a (Image): Image A.
        image_b (Image): Image B.
        direction (str, optional): Where image b will be placed relative to a.
            Options include: left, right, above, below. Default: right.
        fill_value (float, optional): Value to fill additional channels with.
            Default: 0.

    Raises:
        ValueError: If illegal direction given.
        ValueError: If direction is left/right and image heights are not
            equal.
        ValueError: If direction is above/below and image widths are not
            equal.

    Returns:
        Image: Stapled Image.
    '''
    direction = direction.lower()
    dirs = ['left', 'right', 'above', 'below']
    if direction not in dirs:
        msg = f'Illegal direction: {direction}. Legal directions: {dirs}.'
        raise ValueError(msg)

    if direction in ['left', 'right']:
        h0 = image_a.height
        h1 = image_b.height
        if h0 != h1:
            msg = f'Image heights must be equal. {h0} != {h1}.'
            raise ValueError(msg)

    elif direction in ['above', 'below']:
        w0 = image_a.width
        w1 = image_b.width
        if w0 != w1:
            msg = f'Image widths must be equal. {w0} != {w1}.'
            raise ValueError(msg)

    # pad images so number of channels are equal
    a = image_a.data
    b = image_b.data

    # needed for one channel images which make (h, w) arrays
    if len(a.shape) < 3:
        a = np.expand_dims(a, axis=2)
    if len(b.shape) < 3:
        b = np.expand_dims(b, axis=2)

    ca = image_a.num_channels
    cb = image_b.num_channels
    if ca != cb:
        w, h, _ = image_a.shape
        c = abs(ca - cb)
        pad = np.ones((h, w, c), dtype=np.float32) * fill_value
        if ca > cb:
            b = np.concatenate([b, pad], axis=2)
        else:
            a = np.concatenate([a, pad], axis=2)

    if direction == 'above':
        data = np.append(b, a, axis=0)
    elif direction == 'below':
        data = np.append(a, b, axis=0)
    elif direction == 'left':
        data = np.append(b, a, axis=1)
    elif direction == 'right':
        data = np.append(a, b, axis=1)

    return Image.from_array(data)


def cut(image, indices, axis='vertical'):
    # (Image, Union[int, list[int]], str) -> Image
    '''
    Splits a given image into two images along a vertical or horizontal axis.

    .. image:: images/cut.png

    Args:
        image (Image): Image to be cut.
        indices (int or list[int]): Indices of where to cut along cross-axis.
        axis (str, optional): Axis to cut along.
            Options include: vertical, horizontal. Default: vertical.

    Raises:
        EnforceError: If image is not an Image instance.
        EnforceError: If indices is not an int or list of ints.
        EnforceError: If illegal axis is given.
        IndexError: If indices contains index that is outside of bounds.

    Returns:
        tuple[Image]: Two Image instances.
    '''
    Enforce(image, 'instance of', Image)

    axis = axis.lower()
    msg = 'Illegal axis: {a}. Legal axes include: {b}.'
    Enforce(axis, 'in', ['vertical', 'horizontal'], message=msg)

    # create indices
    if isinstance(indices, int):
        indices = [indices]

    Enforce(indices, 'instance of', list)
    enf.enforce_homogenous_type(indices)
    Enforce(indices[0], 'instance of', int)  # tyep: ignore

    indices.append(0)
    max_ = image.width
    if axis == 'horizontal':
        max_ = image.height
    indices.append(max_)
    indices = sorted(list(set(indices)))

    max_i = max(indices)
    if max_i > max_:
        msg = f'Index out of bounds. {max_i} > {max_}.'
        raise IndexError(msg)

    min_i = min(indices)
    if min_i < 0:
        msg = f'Index out of bounds. {min_i} < 0.'
        raise IndexError(msg)
    # --------------------------------------------------------------------------

    output = []
    for j, i in enumerate(indices):
        if j == 0:
            continue
        prev = indices[j - 1]
        if axis == 'vertical':
            img = image[prev:i, :, :]
        else:
            img = image[:, prev:i, :]
        output.append(img)
    return output


def chop(image, channel='a', mode='vertical-horizontal'):
    # type: (Image, str, str) -> dict[tuple[int, int], Image]
    '''
    Chops up a given image into smaller images that bound single contiguous
    objects within a given channel.

    .. image:: images/chop_example.png

    Chop has the following modes:

    .. image:: images/chop_modes.png

    Args:
        image (Image): Image instance.
        channel (str, optional): Channel to chop image by. Default: 'a'.
        mode (str, optional): The type and order of cuts to ber performed.
            Default: vertical-horizontal. Options include:

                * vertical - Make only vertical cuts along the width axis.
                * horizontal - Make only horizontal cuts along the height axis.
                * vertical-horizontal - Cut along the width axis first and then
                    the height axis of each resulting segement.
                * horizontal-vertical - Cut along the height axis first and then
                    the width axis of each resulting segement.

    Raises:
        EnforceError: If image is not an Image or NDArray.
        EnforceError: If channel is not in image channels.
        EnforceError: If illegal mode given.

    Returns:
        dict: Dictionary of form (width, height): Image.
    '''
    # enforce image
    if isinstance(image, np.ndarray):
        image = Image.from_array(image)
    Enforce(image, 'instance of', Image)

    # enforce channel
    msg = '{a} is not a valid channel. Channels include: {b}.'
    Enforce(channel, 'in', image.channels, message=msg)

    # enforce mode
    modes = [
        'vertical', 'horizontal', 'vertical-horizontal', 'horizontal-vertical'
    ]
    msg = '{a} is not a legal mode. Legal modes include: {b}.'
    Enforce(mode, 'in', modes, message=msg)
    # --------------------------------------------------------------------------

    def get_bounding_boxes(image):
        # type: (Image) -> list[list[tuple[int, int]]]
        hgt = image.height
        img = image.to_bit_depth(BitDepth.UINT8).data
        contours = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        items = contours[0] if len(contours) == 2 else contours[1]

        output = []
        for item in items:
            x, y, w, h = cv2.boundingRect(item)
            output.append([
                (x, hgt - (y + h)),
                (x + w, hgt - y),
            ])
        return output

    def get_indices(image, axis):
        # type: (Image, str) -> list[int]
        # find indices of bbox edges across a single axis
        bboxes = get_bounding_boxes(image)
        indices = set()  # type: Any
        if axis == 'vertical':
            for bbox in bboxes:
                indices.add(bbox[0][0])
                indices.add(bbox[1][0])
        else:
            for bbox in bboxes:
                indices.add(bbox[0][1])
                indices.add(bbox[1][1])
        indices = list(indices)

        max_ = image.width if axis == 'vertical' else image.height
        indices = filter(lambda x: x not in [0, max_], indices)
        indices = sorted(indices)
        indices.insert(0, 0)
        indices.append(max_)

        output = []
        for i in indices:
            if i not in output:
                output.append(i)
        return output
    # ------------------------------------------------------------------------

    # get axes
    axes = mode.split('-')
    a0 = axes[0]

    # get indices along first axis
    indices = get_indices(image[:, :, channel], a0)
    segments = cut(image, indices, axis=a0)

    # if only one axis return segments
    output = {}
    if len(axes) == 1:
        if mode == 'vertical':
            output = {(x, 0): item for x, item in enumerate(segments)}
        else:
            yl = len(segments) - 1
            output = {(0, yl - y): item for y, item in enumerate(segments)}
        return output

    # otherwise chop each segement according to its edge detect indices
    a1 = axes[1]
    xl = len(segments) - 1
    for x, segment in enumerate(segments):
        indices = get_indices(segment[:, :, channel], a1)
        images = cut(segment, indices, axis=a1)
        yl = len(images) - 1
        for y, image in enumerate(images):
            if a0 == 'vertical':
                output[(x, yl - y)] = image
            else:
                output[(y, xl - x)] = image
    return output


def warp(
    image,
    angle=0,
    translate_x=0,
    translate_y=0,
    scale=1,
    inverse=False,
    matrix=None
):
    # type: (Image, float, float, float, float, bool, OptArray) -> Image
    '''
    Warp image by given parameters or warp matrix.
    If matrix is given, parameters are ignored.

    Args:
        image (Image): Image.
        angle (float, optional): Rotation in degrees. Default: 0.
        translate_x (float, optional): X translation. Default: 0.
        translate_y (float, optional): Y translation. Default: 0.
        scale (float, optional): Scale factor. Default: 1.
        inverse (bool, optional): Inverse transformation. Default: False.
        matrix (numpy.ndarray, optional): Warp matrix. Default: None.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If matrix is not None or an instance of np.ndarray.
        EnforceError: If matrix is not a 3 x 3 matrix.

    Returns:
        Image: Warped image.
    '''
    Enforce(image, 'instance of', Image)

    if matrix is None:
        matrix = get_warp_matrix(angle, translate_x, translate_y, scale)
    Enforce(matrix, 'instance of', np.ndarray)
    Enforce(matrix.shape, '==', (3, 3))
    # --------------------------------------------------------------------------

    # convert to float
    bit_depth = image.bit_depth
    channels = image.channels
    array = image.to_bit_depth(BitDepth.FLOAT32).data
    flag = cv2.WARP_INVERSE_MAP if inverse else 0
    array = cv2.warpPerspective(
        array, matrix, dsize=image.width_and_height, flags=flag
    )

    # convert to original bit depth
    output = Image.from_array(array).to_bit_depth(bit_depth)

    # set original channel names
    output = output.set_channels(channels)
    return output


def get_warp_matrix(angle=0, translate_x=0, translate_y=0, scale=1):
    # type: (float, float, float, float) -> np.ndarray
    '''
    Create 3 x 3 warp matrix.

    Args:
        angle (float, optional): Rotation in degrees. Default: 0.
        translate_x (float, optional): X translation. Default: 0.
        translate_y (float, optional): Y translation. Default: 0.
        scale (float, optional): Scale factor. Default: 1.

    Returns:
        ndarray: 3 x 3 warp matrix.
    '''
    t = np.radians(angle)
    x = translate_x
    y = translate_y
    s = scale
    return np.array([
        [math.cos(t) * s, -math.sin(t) * s, x],
        [math.sin(t) * s, math.cos(t) * s, y],
        [0, 0, 1]
    ])
