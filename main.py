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
  python main.py
'''

import pygame
import time

from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from scene import TEST_SCENE
from tracer import PathTracer


def entry_point():
  # Initialize pygame and create screen
  pygame.init()
  screen_width = 120
  screen_height = 100
  screen = pygame.display.set_mode((screen_width, screen_height))

  # Initialize Path Tracer
  path_tracer = PathTracer(screen_width, screen_height, TEST_SCENE, 1, 1)

  # Variable to keep the main loop running
  running = True

  # Initialize performance counter
  start_time = time.perf_counter()
  last_time = start_time

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
      curr = path_tracer.current_iteration
      last_time = time.perf_counter()
      pygame.display.set_caption(f"Iterations: {curr}, \
            Iteration Time: {(last_time-start_time)/curr:.5f} sec, \
            It/sec: {curr/(last_time-start_time):.5f}")

    # Get buffer and format for presentation
    buffer = path_tracer.buffer * 255.0

    # Blit frame buffer to screen and present
    pygame.surfarray.blit_array(screen, buffer)
    pygame.display.update()


if __name__ == "__main__":
  entry_point()
