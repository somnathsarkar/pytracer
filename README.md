# pytracer

This is a simple path tracer written in pure python for the CPU. It supports the following features:
1. Monte Carlo path tracing with Importance Sampling
2. Parallel execution with the `multiprocessing` module
3. Lambertian diffuse surfaces
4. Emissive materials
5. Save path tracer state and resume remaining iterations later
6. Live execution in `pygame` window or console

## Environment

You can use `conda` to setup the environment using the included `environment.yml` file. The following commands will create the environment and run the included example.

```
conda env create -f environment.yml
conda activate pytracer-env
pip install .
cd examples/window
python main.py
```

## License Information

This work is made available under the [MIT License](LICENCE.txt)