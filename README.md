# DINEOF_GPU_version
## :star: **THis is the GPU_version based on my previous DINEOF preject.**
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
- ### :rocket:Speedup: The GPU version is at least 5 times faster than the CPU version.

## Usage
- change `eof_multi` to `eof_multi_GPU`; `eof_core` to `eof_core_torch` in DINEOF_MAIN_GPU.py

## Conclusion
The GPU version provides a significant speedup over the CPU version without compromising accuracy, making it a suitable choice for large-scale data reconstruction tasks.

## Contact
If you have any questions, feel free to contact: 'zhouweichen1992110@outlook.com'

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgments
This code is **based on the MATLAB scripts publicly provided by **shepherdmeng** [MATLAB code](https://github.com/aida-alvera/DINEOF/tree/master/Scripts/Matlab](https://github.com/shepherdmeng/Image-Data-Reconstruction-with-DINEOF)). It also references the methodologies outlined in:
- Beckers, J.M., Rixen, M., 2003. *EOF calculations and data filling from incomplete oceanographic datasets*. Journal of Atmospheric & Oceanic Technology, 20.
- Alvera-Azcárate, A., et al., 2007. *Multivariate reconstruction of missing data in sea surface temperature, chlorophyll, and wind satellite fields*. Journal of Geophysical Research, 112.
