# Radar Data Smoothing using Discrete Cosine Transform

A fast spectral domain algorithm for smoothing radar data in polar coordinates.

This repository implements the method described in *[Paper Title TBD]* (submitted to AMT). By operating in the spectral domain using the Discrete Cosine Transform (DCT), this method effectively handles the varying physical width of azimuth beams ("pie slice" distortion) with **$O(N \log N)$ complexity**, achieving up to **800x speedup** over traditional spatial convolution.

## Features
- **Fast**: Spectral domain implementation using FFT.
- **Accurate**: Adapts azimuth kernel width to maintain constant physical smoothing width across ranges.
- **Robust**: Handles missing data (NaNs) via efficient interpolation.
- **Flexible**: Analytical kernels for Boxcar, Gaussian, Savitzky-Golay, and Hanning windows.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
import dct_smoothing

# Example: (360 rays, 1000 gates)
data = np.random.rand(360, 1000) 
az_res_deg = 1.0
width_pixels = 5.0  # Smooth over 5 pixels (physical equivalent)

smoothed = dct_smoothing.apply_polar_dct_smoothing(
    data, 
    az_res_deg=az_res_deg, 
    width_pixels=width_pixels, 
    kernel_type='boxcar'
)
```

## Reproducibility

To reproduce the figures from the paper using synthetic data:
1. Ensure dependencies are installed.
2. Run the notebook `demo_paper_figs.ipynb`.

## Citation

If you use this code, please cite:

```bibtex
@article{YourName202x,
  title={Radar Data Smoothing using the Discrete Cosine Transform},
  author={Your Name and Co-authors},
  journal={Atmospheric Measurement Techniques},
  year={202x}
}
```

## License

MIT License. See `LICENSE` file for details.
