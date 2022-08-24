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

This script runs the path tracer in console and outputs to a file.

  Usage:

  conda env create -f environment.yml
  conda activate pytracer-env
  pip install .
  cd examples/console
  python main.py
'''
import time
import os
import pickle as pkl

import numpy as np

from pytracer.scene import CORNELL_BOX
from pytracer.tracer import PathTracer


def entry_point():
  screen_width = 300
  screen_height = 300

  # Initialize Path Tracer
  tracer_path = "tmp/tracer.pkl"
  buffer_path = "tmp/buffer.npy"
  if os.path.exists(tracer_path):
    with open(tracer_path, "rb") as f:
      path_tracer = pkl.load(f)
  else:
    path_tracer = PathTracer(screen_width, screen_height, CORNELL_BOX, 8, 100,
                             1, 1)

  # Initialize performance counter
  start_time = time.perf_counter()
  last_time = start_time

  # Main loop
  while path_tracer.current_iteration < path_tracer.num_iterations:
    # If iterations aren't complete, run the next iteration
    path_tracer.next_iteration()
    # Update performance counter
    curr = path_tracer.current_iteration
    tot = path_tracer.num_iterations
    last_time = time.perf_counter()
    update_str = (f"Iterations: {curr},"
                  f"Iteration Time: {(last_time-start_time)/curr:.2f}s,"
                  f"It/sec: {curr/(last_time-start_time):.2f},"
                  f"Remaining: {(last_time-start_time)*(tot-curr)/curr:.2f}s,"
                  f"Elapsed: {(last_time-start_time):.2f}s")
    print(update_str)
    # Save tracer state
    path_tracer.save_state(tracer_path)

  # Save output buffer to file
  print("Finished, saving output to file")
  np.save(buffer_path, path_tracer.denoise_buffer)


if __name__ == "__main__":
  entry_point()
