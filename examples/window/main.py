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
'''Entry point for pytracer

This script creates a pygame window and displays the test scene.

  Usage:

  conda env create -f environment.yml
  conda activate pytracer-env
  pip install .
  cd examples/window
  python main.py
'''
import time
import os
import pickle as pkl

from pytracer.scene import CORNELL_BOX
from pytracer.tracer import PathTracer


def entry_point():
  # Initialize pygame and create screen
  # pygame performs setup on import, we avoid this for multiple processes
  import pygame
  from pygame.locals import (
      K_ESCAPE,
      KEYDOWN,
      QUIT,
  )
  pygame.init()
  screen_width = 300
  screen_height = 300
  screen = pygame.display.set_mode((screen_width, screen_height))

  # Initialize Path Tracer
  tracer_path = "tmp/tracer.pkl"
  if os.path.exists(tracer_path):
    with open(tracer_path, "rb") as f:
      path_tracer = pkl.load(f)
  else:
    path_tracer = PathTracer(screen_width, screen_height, CORNELL_BOX, 8, 100,
                             1, 1)

  # Variable to keep the main loop running
  running = True

  # Initialize performance counter
  start_time = time.perf_counter()
  last_time = start_time
  start_iter = path_tracer.current_iteration

  # Main loop
  while running:
    # Process events
    for event in pygame.event.get():
      if event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
      elif event.type == QUIT:
        running = False

    # Update Path Tracer
    if path_tracer.current_iteration < path_tracer.num_iterations:
      # If iterations aren't complete, run the next iteration
      path_tracer.next_iteration()
      # Update performance counter
      curr_iter = path_tracer.current_iteration
      iters = curr_iter - start_iter
      tot = path_tracer.num_iterations
      last_time = time.perf_counter()
      pygame.display.set_caption(
          f"Iterations: {curr_iter},"
          f"Iteration Time: {(last_time-start_time)/iters:.2f}s,"
          f"It/sec: {iters/(last_time-start_time):.2f},"
          f"Remaining: {(last_time-start_time)*(tot-curr_iter)/iters:.2f}s,"
          f"Elapsed: {(last_time-start_time):.2f}s")
      # Save tracer state
      path_tracer.save_state(tracer_path)

    # Get buffer and format for presentation
    buffer = path_tracer.denoise_buffer * 255.0

    # Blit frame buffer to screen and present
    pygame.surfarray.blit_array(screen, buffer)
    pygame.display.update()


if __name__ == "__main__":
  entry_point()
