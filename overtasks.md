## Tasks to complete overnight
So long as it won't comprismise the status of the project, you have free reign to provide some fixes:

- [x] Make sure you're using the correct data type such as `float32` instead of `float64`
      → obs_space and get_observation() already using float32. Also fixed obs bounds:
        pole angle widened from ±0.41887 rad to ±π so the physics sim can return any
        angle during a 100ms step without tripping SB3's observation bounds check.
- [x] Assert that additional dependencies like tensorboard get installed (for examples only)
      → Added python3-tensorboard to package.xml exec_depend. train_sb3.py now
        checks for tensorboard at startup and exits with a clear install message if missing.
- [x] Is there any acceleration we can do with PyTorch?
      → Added --device flag to train_sb3.py (default: "auto" — SB3 picks CUDA if
        available, otherwise CPU). Startup now prints which device is resolved.
        Run with --device cuda to force GPU, --device cpu to force CPU.
