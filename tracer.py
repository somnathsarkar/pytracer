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
'''CPU-based path tracer

  The raygen shader launches one ray for each screen pixel and tests it against
  all triangles in the scene. Each ray trace returns a payload containing the
  pixel color. The colors for each pixel on screen is returned in a buffer by
  the raygen shader.

  Usage:

  import tracer
  buffer = tracer.raygen_shader(SCREEN_WIDTH,SCREEN_HEIGHT,scene)
  # buffer is a SCREEN_WIDTH x SCREEN_HEIGHT x 3 
  # float32 array containing RGB data in the range 0-1
'''

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from camera import view_to_projection_matrix
from scene import Triangle, Ray, Scene
from vector import vec3_to_direction, vec3_to_position, position_to_vec3

EPSILON = 1e-3


@dataclass
class Payload:
  color: np.ndarray


def _ray_triangle_intersection(
    ray: Ray, tri: Triangle) -> Tuple[bool, Optional[np.ndarray]]:
  # ray.position + t * ray.direction = b0*tri.verts[0] + b1*tri.verts[1] + (1-b0-b1)*tri.verts[2]
  # ray.direction * t + (tri.verts[2]-tri.verts[0]) * b0 + (tri.verts[2] - tri.verts[1]) * b1
  # = tri.verts[2] - ray.position
  A = np.vstack((ray.direction, tri.vertices[2] - tri.vertices[0],
                 tri.vertices[2] - tri.vertices[1])).T
  b = tri.vertices[2] - ray.position
  # Solve Ax-b = 0
  singular_error: bool = False
  x = np.linalg.solve(A, b)
  if singular_error or np.any(x < -EPSILON) or np.any(x[1:] > 1 + EPSILON) or (
      1 - x[1] - x[2]) < -EPSILON or (1 - x[1] - x[2]) > 1 + EPSILON:
    return (False, None)
  return (True, x)


def _trace_ray(ray: Ray, scene: Scene) -> Payload:
  payload = Payload(np.array([0.0, 0.0, 0.0]))
  for instance in scene.instances:
    for tri in instance.mesh.tris:
      # Convert triangle vertices into 4x3 matrix with columns as position vectors
      tri_verts = np.hstack(
          [vec3_to_position(tri.vertices[i]).reshape(-1, 1) for i in range(3)])
      transformed_vertices_position = instance.transform @ tri_verts
      transformed_vertices = np.vstack([
          position_to_vec3(transformed_vertices_position[:, i])
          for i in range(3)
      ])
      tri_normal = vec3_to_direction(tri.normal)
      transformed_normal = instance.transform @ tri_normal
      transformed_tri = Triangle(transformed_vertices, transformed_normal)
      intersected, _ = _ray_triangle_intersection(ray, transformed_tri)
      if intersected:
        payload.color = np.array([0.0, 1.0, 0.0])
  return payload


def _unsafe_normalize(vec: npt.NDArray[np.float64]):
  mag: float = np.linalg.norm(vec)
  return vec / mag


def raygen_shader(screen_width: int, screen_height: int,
                  scene: Scene) -> np.ndarray:
  buffer: np.ndarray = np.zeros((screen_width, screen_height, 3),
                                dtype=np.float32)
  screen_size = np.array((screen_width, screen_height))
  # Calculate screen pos to world pos matrix
  # World to Screen = Proj * View * world_pos
  # Screen to World = Inverse View * Inverse Proj * screen_pos
  view_to_world_matrix = scene.main_camera.transform
  proj_to_view_matrix = np.linalg.inv(
      view_to_projection_matrix(scene.main_camera))
  proj_to_world_matrix = view_to_world_matrix @ proj_to_view_matrix
  camera_world_pos = position_to_vec3(
      view_to_world_matrix @ np.array([0.0, 0.0, 0.0, 1.0]))
  for row_i in range(screen_height):
    for col_i in range(screen_width):
      screen_pixel = np.array((col_i, row_i), dtype=np.float32)
      screen_pos_2d = 2.0 * (screen_pixel / screen_size) - 1.0
      screen_pos = np.array([screen_pos_2d[0], screen_pos_2d[1], 0.0, 1.0])
      world_ray_origin = position_to_vec3(proj_to_world_matrix @ screen_pos)
      world_ray_direction = _unsafe_normalize(world_ray_origin -
                                              camera_world_pos)
      ray: Ray = Ray(world_ray_origin, world_ray_direction)
      payload: Payload = _trace_ray(ray, scene)
      buffer[col_i][row_i] = payload.color
  return buffer
