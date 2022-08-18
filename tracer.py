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
'''CPU-based path tracer implementation
'''

from dataclasses import dataclass
import multiprocessing
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from camera import view_to_projection_matrix
from scene import TEST_SCENE, Triangle, Ray, Scene
from vector import rotate_mat, vec3_to_direction, vec3_to_position, position_to_vec3
from vector_types import Vec2, Vec3, Mat4

EPSILON = 1e-3


@dataclass
class Payload:
  color: Vec3


class PathTracer(object):
  """Iterative Path Tracer

  Usage:

  import tracer
  path_tracer = tracer.PathTracer()
  for i in range(path_tracer.num_iterations):
    path_tracer.next_iteration()
    buffer = path_tracer.buffer
  # buffer is a SCREEN_WIDTH x SCREEN_HEIGHT x 3
  # float32 array containing RGB data in the range 0-1
  """

  def __init__(self,
               screen_width: int,
               screen_height: int,
               scene: Scene = TEST_SCENE,
               num_processes: int = 1,
               num_iterations: int = 1,
               num_samples: int = 1,
               max_depth: int = 2):
    self.screen_width = screen_width
    self.screen_height = screen_height
    self.scene = scene
    self.num_processes = num_processes
    self.num_iterations = num_iterations
    self.num_samples = num_samples
    self.max_depth = max_depth
    self.current_iteration = 0
    self.buffer = np.zeros((screen_width, screen_height, 3), dtype=np.float32)
    self.batch_size = max(
        (self.screen_width * self.screen_height) // num_iterations, 1)

    # Setup work
    self.screen_positions = np.dstack(
        np.meshgrid(np.arange(screen_width),
                    np.arange(screen_height))).reshape(-1, 2)

  @staticmethod
  def _ray_triangle_intersection(
      ray: Ray, tri: Triangle) -> Tuple[bool, Optional[np.ndarray]]:
    # ray.position + t*ray.direction = b0*tri.verts[0] + b1*tri.verts[1]
    # + (1-b0-b1)*tri.verts[2]
    # ray.direction*t + (tri.verts[2]-tri.verts[0])*b0
    # + (tri.verts[2]-tri.verts[1])*b1
    # = tri.verts[2] - ray.position
    solve_a = np.vstack((ray.direction, tri.vertices[2] - tri.vertices[0],
                         tri.vertices[2] - tri.vertices[1])).T
    solve_b = tri.vertices[2] - ray.position
    # Solve Ax-b = 0
    singular_error: bool = False
    x = np.linalg.solve(solve_a, solve_b)
    if singular_error or np.any(x < -EPSILON) or np.any(
        x[1:] > 1 + EPSILON) or (1 - x[1] - x[2]) < -EPSILON or (
            1 - x[1] - x[2]) > 1 + EPSILON or x[0] < EPSILON:
      return (False, None)
    return (True, x)

  @staticmethod
  def _get_basis_transform(normal: Vec3) -> Mat4:
    random_cross = PathTracer._unsafe_normalize(np.random.rand(3))
    tangent = np.cross(random_cross, normal)
    bitangent = np.cross(tangent, normal)
    w = np.array([0.0, 0.0, 0.0])
    mat = np.vstack([
        vec3_to_direction(basis_vector)
        for basis_vector in (normal, tangent, bitangent, w)
    ])
    mat[3, 3] = 1.0
    return mat

  def _trace_ray(self, ray: Ray, depth: int, ignore_instance: int,
                 ignore_tri: int) -> Payload:
    payload = Payload(np.array([0.0, 0.0, 0.0]))
    found_hit = False
    for instance_i, instance in enumerate(self.scene.instances):
      for tri_i, tri in enumerate(instance.mesh.tris):
        if ignore_instance == instance_i and tri_i == ignore_tri:
          continue
        # Convert triangle vertices into 4x3 matrix with position vector columns
        tri_verts = np.hstack([
            vec3_to_position(tri.vertices[i]).reshape(-1, 1) for i in range(3)
        ])
        transformed_vertices_position = instance.transform @ tri_verts
        transformed_vertices = np.vstack([
            position_to_vec3(transformed_vertices_position[:, i])
            for i in range(3)
        ])
        tri_normal = vec3_to_direction(tri.normal)
        transformed_normal = (instance.transform @ tri_normal)[:3]
        transformed_tri = Triangle(transformed_vertices, transformed_normal)
        intersected, intersect_params = PathTracer._ray_triangle_intersection(
            ray, transformed_tri)
        if intersected:
          intersect_albedo = instance.material.albedo
          payload.color = instance.material.emission
          intersect_t = intersect_params[0]
          if depth > 0:
            sample_origin = ray.position + intersect_t * ray.direction
            inv_intersect_basis = PathTracer._get_basis_transform(
                transformed_normal).T
            for _ in range(self.num_samples):
              azimuth = np.random.rand() * np.pi - np.pi / 2.0
              inclination = np.arcsin(np.random.rand() * 2.0 - 1.0)
              random_ray_transform = rotate_mat(
                  np.array([0.0, inclination, azimuth]))
              sample_ray_direction = inv_intersect_basis @ random_ray_transform @ np.array(
                  [1.0, 0.0, 0.0, 0.0])
              sample_ray_direction = sample_ray_direction[:3]
              sample_ray = Ray(sample_origin, sample_ray_direction)
              sample_payload = self._trace_ray(sample_ray, depth - 1,
                                               instance_i, tri_i)
              payload.color = (payload.color) + (4 * sample_payload.color *
                                                 intersect_albedo /
                                                 self.num_samples)
          falloff = max(intersect_t * intersect_t, 1.0)
          payload.color = payload.color / falloff
          found_hit = True
          break
      if found_hit:
        break
    return payload

  def _ray_worker(self, screen_pos: Vec2) -> Vec3:
    screen_size = np.array((self.screen_width, self.screen_height))
    # Calculate screen pos to world pos matrix
    # World to Screen = Proj * View * world_pos
    # Screen to World = Inverse View * Inverse Proj * screen_pos
    view_to_world_matrix = self.scene.main_camera.transform
    proj_to_view_matrix = np.linalg.inv(
        view_to_projection_matrix(self.scene.main_camera))
    proj_to_world_matrix = view_to_world_matrix @ proj_to_view_matrix
    camera_world_pos = position_to_vec3(
        view_to_world_matrix @ np.array([0.0, 0.0, 0.0, 1.0]))
    screen_x, screen_y = screen_pos
    screen_pixel = np.array((screen_x, screen_y), dtype=np.float32)
    screen_pos_2d = 2.0 * (screen_pixel / screen_size) - 1.0
    screen_pos = np.array([screen_pos_2d[0], screen_pos_2d[1], 0.0, 1.0])
    world_ray_origin = position_to_vec3(proj_to_world_matrix @ screen_pos)
    world_ray_direction = PathTracer._unsafe_normalize(world_ray_origin -
                                                       camera_world_pos)
    ray: Ray = Ray(world_ray_origin, world_ray_direction)
    payload: Payload = self._trace_ray(ray, self.max_depth, -1, -1)
    return payload.color

  @staticmethod
  def _unsafe_normalize(
      vec: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    mag: float = np.linalg.norm(vec)
    return vec / mag

  def next_iteration(self):
    assert self.current_iteration < self.num_iterations
    batch_start = self.current_iteration * self.batch_size
    batch_end = batch_start + self.batch_size
    batch = self.screen_positions[batch_start:batch_end]
    pool = multiprocessing.Pool(processes=self.num_processes)
    payload_colors = pool.map(self._ray_worker, batch)
    self.buffer[batch[:, 0], batch[:, 1]] = np.minimum(payload_colors, 1.0)
    self.current_iteration += 1
