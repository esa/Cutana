#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Preprocessing function for fits data"""

import numpy as np

from .constants import JANSKY_AB_ZEROPONT


def apply_flux_conversion(config, img: np.ndarray, header) -> np.ndarray:
    """Applies flux conversion function for fits data

    Args:
        config (DotMap): Configuration object with flux conversion settings
        img (numpy.ndarray): Image data
        header (astropy.io.fits.Header): FITS header

    Returns:
        numpy.ndarray: Flux-converted image data

    Raises:
        ValueError: If zeropoint keyword is not found in header
    """

    if config.apply_flux_conversion:

        if config.user_flux_conversion_function:
            img = config.user_flux_conversion_function(img, header)
        else:
            # get zeropoint from image header via the cfg dict
            # first get zeropoint header keyword from config
            zeropoint_keyword = config.flux_conversion_keywords.AB_zeropoint
            zeropoint = header.get(zeropoint_keyword, None)

            if zeropoint is None:
                raise ValueError(
                    f"Zeropoint keyword '{zeropoint_keyword}' not found in FITS header"
                )

            img = convert_mosaic_to_flux(img, zeropoint)

    return img


def convert_mosaic_to_flux(img: np.ndarray, AB_zeropoint: float) -> np.ndarray:
    """
    Convert mosaic pixel values to flux using the given zeropoint.
    For a definition to get the conversion see
    https://en.wikipedia.org/wiki/AB_magnitude

    Parameters:
    -----------
    img : numpy.ndarray
        Image data
    AB_zeropoint : float
        Zeropoint value to convert to AB magnitudes

    Returns:
    --------
    numpy.ndarray
        Flux values in Jansky
    """
    # Definition:
    # magab = -2.5 * np.log10(im) + zp
    # f [3631 jansky] = 10**(-0.4*(magab)) ; as  1/2.5 = 0.4
    # to get flux in jansky multiply with the factor
    # jansky ab zp can be taken from astropy or the literature
    flux = img * 10 ** (-0.4 * AB_zeropoint) * JANSKY_AB_ZEROPONT
    return flux
