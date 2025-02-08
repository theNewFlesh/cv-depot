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
