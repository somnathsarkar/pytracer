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
"""Utilities for perspective cameras
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import math


@dataclass
class Camera:
  transform: np.ndarray
  fovx: float


def view_to_projection_matrix(camera: Camera) -> npt.NDArray[np.float64]:
  near: float = 0.01
  far: float = 100.0
  aspect_ratio: float = 16.0 / 9.0
  hcot: float = 1.0 / math.tan(math.radians(camera.fovx / 2.0))
  mat: npt.NDArray[np.float64] = np.array(
      [[hcot, 0.0, 0.0, 0.0], [0.0, aspect_ratio * hcot, 0.0, 0.0],
       [0.0, 0.0, far / (far - near), (-near * far) / (far - near)],
       [0.0, 0.0, 1.0, 0.0]])
  return mat
