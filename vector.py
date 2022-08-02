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
'''Collection of common vector utilities
'''

import numpy as np
from vector_types import Vec3, Vec4


# For a single vec3 (x,y,z) return (x,y,z,0)
def vec3_to_direction(vec: Vec3) -> Vec4:
  return np.pad(vec, (0, 1))


# For a single vec3 (x,y,z) return (x,y,z,1)
def vec3_to_position(vec: Vec3) -> Vec4:
  return np.pad(vec, (0, 1), constant_values=1)


# Return position vector after perspective divide
def position_to_vec3(vec: Vec4) -> Vec3:
  return vec[:3] / vec[3]
