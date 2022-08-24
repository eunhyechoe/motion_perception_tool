# Copyright (C) 2022 Aravind Battaje
# Email: aravind@oxidification.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser Public License for more details.
#
# You should have received a copy of the GNU Lesser Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def gabor_kernel_separate(frequency, theta=0, n_stds=3):
    def _sigma_prefactor(bandwidth):
        b = bandwidth
        # See http://www.cs.rug.nl/~imaging/simplecell.html
        return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
            (2.0 ** b + 1) / (2.0 ** b - 1)
        
    sigma = _sigma_prefactor(1) / frequency

    width = np.ceil(max(np.abs(n_stds * sigma * np.cos(theta)), np.abs(n_stds * sigma * np.sin(theta)), 1))

    y = np.arange(-width, width + 1)
    x = np.arange(-width, width + 1)
    
    rotxx = x * np.cos(theta)
    rotyx = -x * np.sin(theta)
    rotxy = y * np.sin(theta)
    rotyy = y * np.cos(theta)

    kernel_x = np.zeros(x.shape, dtype=complex)
    kernel_y = np.zeros(y.shape, dtype=complex)

    kernel_x[:] = np.exp(-0.5 * (rotxx ** 2 / sigma ** 2 + rotyx ** 2 / sigma ** 2))
    kernel_y[:] = np.exp(-0.5 * (rotxy ** 2 / sigma ** 2 + rotyy ** 2 / sigma ** 2))

    kernel_x /= np.sqrt(2 * np.pi) * sigma
    kernel_y /= np.sqrt(2 * np.pi) * sigma

    kernel_x *= np.exp(1j * (2 * np.pi * frequency * rotxx))
    kernel_y *= np.exp(1j * (2 * np.pi * frequency * rotxy))

    return kernel_x, kernel_y

def gabor_kernel_skimage(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None,
                 n_stds=3, offset=0):
    """Return complex 2D Gabor filter kernel.

    Exact implementation from Scikit-Image. See documentation at
    https://github.com/scikit-image/scikit-image/blob/v0.18.0/skimage/filters/_gabor.py#L16-L95
    """

    # Copied instead of using from skimage because this is used as reference
    # for separating the kernel to 1-D kernels (more efficient convs)

    def _sigma_prefactor(bandwidth):
        b = bandwidth
        # See http://www.cs.rug.nl/~imaging/simplecell.html
        return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
            (2.0 ** b + 1) / (2.0 ** b - 1)
        
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    g = np.zeros(y.shape, dtype=complex)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    return g