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
import numpy as np
import time

from pygame.locals import (
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from scene import TEST_SCENE
from tracer import raygen_shader


def entry_point():
  # Initialize pygame and create screen
  pygame.init()
  screen_width = 120
  screen_height = 100
  screen = pygame.display.set_mode((screen_width, screen_height))

  # Variable to keep the main loop running
  running = True

  # Write frame buffer
  def get_frame() -> np.array:
    arr = raygen_shader(screen_width, screen_height, TEST_SCENE) * 255.0
    return arr

  # Initialize performance counter
  perf_update_interval = 1
  perf_time = time.perf_counter()
  frames = 0

  # Main loop
  while running:
    # Process events
    for event in pygame.event.get():
      if event.type == KEYDOWN:
        if event.key == K_ESCAPE:
          running = False
      elif event.type == QUIT:
        running = False

    # Blit frame buffer to screen and present
    pygame.surfarray.blit_array(screen, get_frame())
    pygame.display.update()

    # Update performance counter
    frames += 1
    new_perf_time = time.perf_counter()
    delta_time = new_perf_time - perf_time
    perf_time = new_perf_time
    if (frames % perf_update_interval) == 0:
      pygame.display.set_caption(
          f"Frames: {frames}, Frame Time: {delta_time:.5f} secs, \
          FPS: {1.0 / delta_time:.5f}")


if __name__ == "__main__":
  entry_point()
