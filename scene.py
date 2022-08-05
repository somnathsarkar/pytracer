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
'''Classes for representing scene information
'''

from dataclasses import dataclass
import numpy as np
from typing import List
from camera import Camera


@dataclass
class Triangle:
  vertices: np.ndarray
  normal: np.ndarray


@dataclass
class Mesh:
  tris: List[Triangle]


@dataclass
class Ray:
  position: np.ndarray
  direction: np.ndarray


@dataclass
class Material:
  albedo: np.ndarray


@dataclass
class Instance:
  transform: np.ndarray
  mesh: Mesh
  material: Material


@dataclass
class Scene:
  instances: List[Instance]
  main_camera: Camera


PLANE = Mesh([
    Triangle(
        np.array([(-1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)]),
        np.array([0.0, 0.0, -1.0])),
    Triangle(
        np.array([(-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]),
        np.array([0.0, 0.0, -1.0]))
])

RED_MATERIAL = Material(np.array([1.0, 0.0, 0.0]))
GREEN_MATERIAL = Material(np.array([0.0, 1.0, 0.0]))
BLUE_MATERIAL = Material(np.array([0.0, 0.0, 1.0]))

_camera_transform = np.diag([1.0, 1.0, 1.0, 1.0])
_camera_transform[2, 3] = -1.0
TEST_SCENE = Scene([Instance(np.eye(4), PLANE, GREEN_MATERIAL)],
                   Camera(_camera_transform, 100.0))
