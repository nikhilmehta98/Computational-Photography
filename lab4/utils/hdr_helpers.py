# -*- coding: utf-8 -*-
""" Implements python port of gsolve.m """

# imports
import numpy as np

# metadata
__author__ = "Jae Yong Lee"
__copyright__ = "Copyright 2019, CS445"
__credits__ = ["Jae Yong Lee"]
__license__ = "None"
__version__ = "1.0.1"
__maintainer__ = "Jae Yong Lee"
__email__ = "lee896@illinois.edu"
__status__ = 'production'

# implementation
def gsolve(Z: np.ndarray, B: np.ndarray, l: int, w) -> (np.ndarray, np.ndarray):
    '''
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system’s response function g as well as the log film irradiance
    values for the observed pixels.

    Arguments:
        Z: N x P array for P pixels in N images
        B: is the log delta t, or log shutter speed, for image j
        l: lambda, the constant that determines smoothness
        w: is the weighting function value for pixel value
    Returns:
        g: solved g value per intensity, of shape 256 x 1
        le: log irradiance for sample pixels of shape P x 1
    '''

    N, P = Z.shape

    n = 256
    A = np.zeros(((N * P) + n + 1, n + P), dtype=np.float32)
    b = np.zeros((A.shape[0], 1))

    k = 0
    # for each pixel
    for i in range(N):
        # for each image
        for j in range(P):
            wij = w(Z[i, j] + 1)
            A[k, Z[i, j]] = wij
            A[k, n + j] = -wij
            b[k, 0] = wij * B[i]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1

    # Include the smoothness equation
    for i in range(n - 2):
        A[k, i] = l * w(i + 1)
        A[k, i + 1] = -2 * l * w(i + 1)
        A[k, i + 2] = l * w(i + 1)
        k += 1
    x = np.linalg.lstsq(A, b)[0]
    g = x[:n, 0]
    lE = x[n:, 0]

    return g, lE
