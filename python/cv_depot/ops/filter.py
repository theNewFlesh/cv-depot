from lunchbox.enforce import Enforce
import cv2
import numpy as np

from cv_depot.core.image import BitDepth, Image
# ------------------------------------------------------------------------------


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
