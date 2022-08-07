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
import vector


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
WHITE_MATERIAL = Material(np.array([1.0, 1.0, 1.0]))

_camera_transform = vector.translate_mat(np.array([0, 0, -1]))
TEST_SCENE = Scene([Instance(np.eye(4), PLANE, GREEN_MATERIAL)],
                   Camera(_camera_transform, 100.0))

_back_transform = vector.translate_mat(np.array([0.0, 0.0, 1.0]))
_back_wall = Instance(_back_transform, PLANE, WHITE_MATERIAL)
_left_transform = vector.translate_mat(np.array(
    [-1.0, 0.0, 0.0])) @ vector.rotate_mat(np.array([0.0, -90.0, 0.0]))
_left_wall = Instance(_left_transform, PLANE, RED_MATERIAL)
_right_transform = vector.translate_mat(np.array(
    [1.0, 0.0, 0.0])) @ vector.rotate_mat(np.array([0.0, 90.0, 0.0]))
_right_wall = Instance(_right_transform, PLANE, GREEN_MATERIAL)
_top_transform = vector.translate_mat(np.array(
    [0.0, -1.0, 0.0])) @ vector.rotate_mat(np.array([-90.0, 0.0, 0.0]))
_top_wall = Instance(_top_transform, PLANE, WHITE_MATERIAL)
_bottom_transform = vector.translate_mat(np.array(
    [0.0, 1.0, 0.0])) @ vector.rotate_mat(np.array([90.0, 0.0, 0.0]))
_bottom_wall = Instance(_bottom_transform, PLANE, WHITE_MATERIAL)
CORNELL_BOX = Scene([
    _back_wall,
    _left_wall,
    _right_wall,
    _top_wall,
    _bottom_wall,
], Camera(_camera_transform, 100.0))
