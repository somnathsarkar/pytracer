# MIT License
#
# Copyright (c) 2022 Somnath Sarkar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Collection of denoisers for path tracer output"""
import numpy as np
import numpy.typing as npt
import scipy.stats


class Denoiser(object):
  """Base Denoiser class
  Returns input buffer with no operations applied"""

  def __init__(self):
    pass

  def denoise(self, buffer: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return buffer


class GaussianFilter(Denoiser):
  """Run a gaussian kernel over input buffer"""

  def __init__(self, kernel_size: int = 3):
    super().__init__()
    self.kernel_size = int(kernel_size)
    assert self.kernel_size > 0 and self.kernel_size % 2 == 1

  def denoise(self, buffer: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    gaussian_1d = scipy.stats.norm.pdf(
        np.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    denoise_buffer = np.zeros_like(buffer)
    for color_i in range(3):
      for i in range(buffer.shape[0]):
        denoise_buffer[i, :, color_i] = np.convolve(buffer[i, :, color_i],
                                                    gaussian_1d, 'same')
      for j in range(buffer.shape[1]):
        denoise_buffer[:, j,
                       color_i] = np.convolve(denoise_buffer[:, j, color_i],
                                              gaussian_1d, 'same')
    return denoise_buffer
