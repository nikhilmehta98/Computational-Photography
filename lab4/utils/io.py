# -*- coding: utf-8 -*-
""" Contains data reading / writing for imageries """

import cv2
import numpy as np

# metadata
__author__ = "Jae Yong Lee"
__copyright__ = "Copyright 2019, CS445"
__credits__ = ["Jae Yong Lee"]
__license__ = "None"
__version__ = "1.0.0"
__maintainer__ = "Jae Yong Lee"
__email__ = "lee896@illinois.edu"
__status__ = 'production'

def write_image(image:np.ndarray, image_path: str):
    '''
    Writes image from image path
    Args:
        image: RGB image of shape H x W x C, with float32 data
        image_path: path to image

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    # read image and convert to RGB
    bgr_image = (image[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    cv2.imwrite(image_path, bgr_image)


def read_image(image_path: str) -> np.ndarray:
    '''
    Reads image from image path, and 
    return floating point RGB image
    
    Args:
        image_path: path to image

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    # read image and convert to RGB
    bgr_image = cv2.imread(image_path)
    rgb_image = bgr_image[:, :, [2, 1, 0]]
    return rgb_image.astype(np.float32) / 255


def read_hdr_image(image_path: str) -> np.ndarray:
    '''
    Reads image from image path, and 
    return HDR floating point RGB image
    
    Args:
        image_path: path to hdr image

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    
    # read image and convert to RGB
    bgr_hdr_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    rgb_hdr_image = bgr_hdr_image[:, :, [2, 1, 0]]
    return rgb_hdr_image.astype(np.float32)

def write_hdr_image(hdr_image: np.ndarray, image_path: str):
    '''
    Write HDR image to given path.
    The path must end with '.hdr' extension
    Args:
        hdr_image: H x W x C float32 HDR image in BGR format.
        image_path: path to hdr image to save

    Returns:
        RGB image of shape H x W x 3 in floating point format
    '''
    assert(image_path.endswith('.hdr'))
    rgb_hdr_image = hdr_image[:, :, [2, 1, 0]]
    cv2.imwrite(image_path, rgb_hdr_image)