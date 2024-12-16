# DINEOF_GPU_version
THis is the torch_gpu_version based on my previous DINEOF preject.
# README

## Overview
This project includes both CPU and GPU implementations of the DINEOF algorithm for data reconstruction. Below are details about the key components and results of testing.

### GPU Implementation
- Core GPU functions are implemented in `utils.py`, specifically in the following functions:
  - `eof_multi_GPU`
  - `eof_core_torch`

### Testing GPU vs. CPU
- The script `cal_error_and_times.py` evaluates the performance of the GPU by :
  - Execution time
  - Reconstruction testError

### Results
- The GPU implementation achieves the same computational accuracy as the CPU version.
- Speedup: The GPU version is at least 5 times faster than the CPU version.

## Usage
- change `eof_multi` to `eof_multi_GPU`; `eof_core` to `eof_core_torch` in DINEOF_MAIN_GPU.py

## Conclusion
The GPU version provides a significant speedup over the CPU version without compromising accuracy, making it a suitable choice for large-scale data reconstruction tasks.
