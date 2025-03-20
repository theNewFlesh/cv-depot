from typing import Any, Optional, Union  # noqa F401
from cv_depot.core.types import AnyColor  # noqa F401

import logging

from lunchbox.enforce import Enforce
from lunchbox.stopwatch import StopWatch
import cv2
import numpy as np

from cv_depot.core.image import BitDepth, Image
from cv_depot.core.color import BasicColor, Color

LOGGER = logging.getLogger(__name__)
# ------------------------------------------------------------------------------


def gamma(image, value=1.0):
    # type: (Image, float) -> Image
    '''
    Apply gamma correction to given image.

    Args:
        image (Image): Image to be modified.
        value (int): Gamma value. Default: 1.0.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If value is less than 0.

    Returns:
        Image: Gamma adjusted image.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(value, '>=', 0)
    # --------------------------------------------------------------------------

    bit_depth = image.bit_depth
    array = image.to_bit_depth(BitDepth.FLOAT32).data
    array = array**(1 / value)
    output = Image.from_array(array).to_bit_depth(bit_depth)
    return output


def canny_edges(image, size=0):
    # type: (Image, int) -> Image
    '''
    Apply a canny edge detection to given image.

    Args:
        image (Image): Image.
        size (int, optional): Amount of dilation applied to result. Default: 0.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If size is not an integer >= 0.

    Returns:
        Image: Edge detected image.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(size, 'instance of', int)
    Enforce(size, '>=', 0)
    # --------------------------------------------------------------------------

    img = image.to_bit_depth(BitDepth.UINT8).data
    img = cv2.Canny(img, 0, 0)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=size)
    output = Image.from_array(img).to_bit_depth(image.bit_depth)
    return output


def tophat(image, amount, kind='open'):
    # type: (Image, int, str) -> Image
    '''
    Apply tophat morphology operation to given image.

    Args:
        image (Image): Image to be modified.
        amount (int): Amount of tophat operation.
        kind (str, optional): Kind of operation to be performed.
            Options include: open, close. Default: open.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If amount is less than 0.
        EnforceError: If kind is not one of: open, close.

    Returns:
        Image: Image with tophat operation applied.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(amount, '>=', 0)
    msg = 'Illegal tophat kind: {a}. Legal tophat kinds: {b}.'
    Enforce(kind, 'in', ['open', 'close'], message=msg)
    # --------------------------------------------------------------------------

    lut = dict(open=cv2.MORPH_CLOSE, close=cv2.MORPH_OPEN)
    opt = lut[kind]
    bit_depth = image.bit_depth
    img = image.to_bit_depth(BitDepth.FLOAT32).data
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    arr = cv2 \
        .morphologyEx(img, opt, kernel, iterations=amount) \
        .astype(np.float32)
    output = Image.from_array(arr).to_bit_depth(bit_depth)
    return output


def linear_lookup(lower=0, upper=1):
    # type: (float, float) -> np.vectorize
    r'''
    Generates a linear lookup table with an upper and lower shoulder.

    .. image:: images/linear_lut.png

    Args:
        lower (float, optional): Lower shoulder value. Default: 0.
        upper (float, optional): Upper shoulder value. Default: 1.

    Returns:
        numpy.vectorize: Anonymous function that applies lut elementwise to a
            given numpy array.
    '''
    lut = lambda x: min(max((x - lower), 0) * (1 / (upper - lower)), 1)
    return np.vectorize(lut)


def linear_smooth(image, blur=3, lower=0, upper=1):
    # type: (Image, int, float, float) -> Image
    '''
    Blur given image then apply linear thresholding it.

    Args:
        image (Image): Image matte to be smoothed.
        blur (int, optional): Size of blur. Default: 3.
        lower (float, optional): Lower shoulder value. Default: 0.
        upper (float, optional): Upper shoulder value. Default: 1.

    Raises:
        EnforceError: If image is not an instance of Image.
        EnforceError: If blur is less than 0.
        EnforceError: If lower or upper is less than 0.
        EnforceError: If lower or upper is greater than 1.
        EnforceError: If lower is greater than upper.

    Returns:
        Image: Smoothed image.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(blur, '>=', 0)
    Enforce(lower, '>=', 0)
    Enforce(lower, '<=', 1)
    Enforce(upper, '>=', 0)
    Enforce(upper, '<=', 1)
    Enforce(lower, '>=', 0)
    msg = 'Lower bound cannot be greater than upper bound. {a} > {b}'
    Enforce(lower, '<=', upper, message=msg)
    # --------------------------------------------------------------------------

    bit_depth = image.bit_depth
    img = image.to_bit_depth(BitDepth.FLOAT32).data
    img = cv2.blur(img, (blur, blur))
    lut = linear_lookup(lower=lower, upper=upper)
    img = lut(img).astype(np.float32)
    output = Image.from_array(img).to_bit_depth(bit_depth)
    return output


def key_exact_color(image, color, channel='a', invert=False):
    # type: (Image, AnyColor, str, bool) -> Image
    '''
    Keys given image according to the color of its pixels values.
    Where that pixel color exactly matches the given color, the mask channel
    will be 1, otherwise it will be 0.

    Args:
        image (Image): Image to be evaluated.
        color (Color or BasicColor): Color to be used for masking.
        channel (str, optional): Mask channel name. Default: a.
        invert (bool, optional): Whether to invert the mask. Default: False.

    Raises:
        EnforceError: If image is not an Image instance.
        EnforceError: If channel is not a string.
        EnforceError: If invert is not a boolean.
        EnforceError: If RGB is not found in image channels.

    Returns:
        Image: Image with mask channel.
    '''
    Enforce(image, 'instance of', Image)
    Enforce(channel, 'instance of', str)
    Enforce(invert, 'instance of', bool)
    # --------------------------------------------------------------------------

    # get color
    if isinstance(color, str):
        color = BasicColor.from_string(color)
    if isinstance(color, BasicColor):
        color = Color.from_basic_color(color)
    clr = color.to_array()

    # determine num channels
    rgb = list('rgb')
    img = image.to_bit_depth(BitDepth.FLOAT32)
    if image.num_channels == 1:
        x = img.data[..., np.newaxis]
        x = np.concatenate([x, x, x], axis=2)
        img = Image.from_array(x)
    else:
        diff = sorted(list(set(rgb).difference(image.channels)))
        msg = f'{diff} not found in image channels. '
        msg += f'Given channels: {image.channels}.'
        Enforce(len(diff), '==', 0, message=msg)

    # create mask
    mask = np.equal(clr, img[:, :, rgb].data)
    mask = np.apply_along_axis(all, 2, mask) \
        .astype(np.float32)[..., np.newaxis]
    if invert:
        mask = -1 * mask + 1

    # add mask to image
    chans = image.channels
    if image.num_channels == 1:
        arr = img[:, :, 'r'].data[..., np.newaxis]
    else:
        chans = list(filter(lambda x: x != channel, img.channels))
        arr = img[:, :, chans].data

    arr = np.concatenate([arr, mask], axis=2)
    output = Image.from_array(arr).to_bit_depth(image.bit_depth)
    output = output.set_channels(chans + [channel])
    return output


def kmeans(
    image,                 # type: Image
    num_centroids=10,      # type: int
    centroids=None,        # type: Optional[list[tuple[int, int, int]]]
    max_iterations=100,    # type: int
    accuracy=1.0,          # type: float
    epochs=10,             # type: int
    seeding='random',      # type: str
    generate_report=False  # type: bool
):                         # type: (...) -> Union[Image, tuple[Image, dict]]
    '''
    Applies k-means to the given image.

    Args:
        image (Image): Image instance.
        num_centroids (int, optional): Number of centroids to use. Default: 10.
        centroids (list, optional): List of triplets. Default: None.
        max_iterations (int, optional): Maximum number of k-means updates
            allowed per centroid. Default: 100.
        accuracy (float, optional): Minimum accuracy required of clusters.
        epochs (int, optional): Number of times algorithm is applied with
            different initial labels. Default: 10.
        seeding (str, optional): How intial centroids are generated. Default:
            random. Options include: random, pp_centers.
        generate_report (bool, optional): If true returns report in addition to
            image. Default: False.

    Raises:
        EnforceError: If image is not an Image instance.
        ValueError: If invalid seeding option is given.

    Returns:
        Image or tuple[Image, dict]: K-means image or K-means image and K-means
            report.
    '''
    Enforce(image, 'instance of', Image)

    stopwatch = StopWatch()
    stopwatch.start()
    # --------------------------------------------------------------------------

    source_bit_depth = image.bit_depth
    data = image.to_bit_depth(BitDepth.FLOAT32)\
        .data.reshape((-1, image.num_channels))

    seed = None
    if seeding == 'random':
        seed = cv2.KMEANS_RANDOM_CENTERS
    elif seeding == 'pp_centers':
        seed = cv2.KMEANS_PP_CENTERS
    else:
        msg = f'{seeding} is not a valid seeding option. Options include: '
        msg += '[random, pp_centers].'
        raise ValueError(msg)

    # terminate centroid updates when max iteration or min accuracy is achieved
    crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER

    if centroids is not None:
        num_centroids = len(centroids)

    compactness, labels, centroids = cv2.kmeans(
        data=data,
        K=num_centroids,
        bestLabels=centroids,
        criteria=(crit, max_iterations, accuracy),
        attempts=epochs,
        flags=seed,
    )  # type: ignore

    centroids_ = centroids  # type: Any
    output = np.float32(centroids_)[labels.flatten()]  # type: ignore
    output = output.reshape(image.data.shape)
    output = Image.from_array(output).to_bit_depth(source_bit_depth)

    if generate_report:
        report = dict(
            compactness=compactness,
            labels=labels,
            centroids=centroids
        )
        stopwatch.stop()
        LOGGER.warning(f'KmeansRuntime: {stopwatch.human_readable_delta}.')
        return output, report

    stopwatch.stop()
    LOGGER.warning(f'Kmeans Runtime: {stopwatch.human_readable_delta}.')
    return output
