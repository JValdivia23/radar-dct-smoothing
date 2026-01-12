"""
Radar Data Smoothing using the Discrete Cosine Transform (DCT).

This module implements a fast spectral domain algorithm for smoothing radar data
in polar coordinates. It handles the non-uniform physical width of azimuth beams
by adapting the smoothing kernel in the spectral domain.

Key features:
- O(N log N) complexity using FFT-based DCT.
- Handles polar geometry (varying azimuth resolution).
- Supports missing data (NaNs) via interpolation.
- Analytical transfer functions for Boxcar, Gaussian, etc.
"""

import numpy as np
import scipy.fft
import scipy.signal
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Union

def interpolate_2d_along_axis(data_in: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Interpolate missing values (NaNs) in a 2D array along a specified axis.

    Parameters
    ----------
    data_in : np.ndarray
        Input 2D array with NaNs.
    axis : int, optional
        Axis along which to interpolate. 
        0 for vertical (columns), 1 for horizontal (rows). Default is 0.

    Returns
    -------
    np.ndarray
        Array with NaNs filled by linear interpolation.
        Remaining NaNs (at edges) are filled with the 1st percentile value.
    """
    data = np.copy(data_in)
    
    if axis == 0:
        data = data.T
        
    # Now we always interpolate along the last axis (rows)
    # data.shape[0] is the number of rows (original columns if axis=0)
    
    x = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        y = data[i, :]
        mask = np.isfinite(y)
        if np.any(mask):
            # Only interpolate if we have at least 2 points, strictly speaking
            # but interp1d works with minimal points if bounds_error=False
            if len(x[mask]) > 1:
                interp_func = interp1d(x[mask], y[mask], bounds_error=False, fill_value=None)
                data[i, ~mask] = interp_func(x[~mask])
    
    if axis == 0:
        data = data.T
        
    # Fill remaining NaNs (e.g., if a whole row/col was NaN or edges)
    if np.all(np.isnan(data)):
        return np.zeros_like(data)
        
    # Use robust value (1st percentile) instead of 0 or meand to avoid bias
    val = np.nanpercentile(data, 1)
    data[np.isnan(data)] = val
    return data

def get_transfer_function(n: int, kernel_type: str, width: float, **kwargs) -> np.ndarray:
    """
    Generates the 1D Discrete Cosine Transform (DCT) Transfer Function H[k].

    Parameters
    ----------
    n : int
        Number of points in the dimension.
    kernel_type : str
        Type of smoothing kernel. Options: 'boxcar', 'gaussian', 'savgol', 'hanning', 'boxcar_discrete'.
    width : float
        Width of the kernel. 
        - For 'boxcar': Full width.
        - For 'gaussian': Equivalent width (approx sqrt(12)*sigma).
    **kwargs : dict
        Additional arguments, e.g., 'poly' order for Savitzky-Golay.

    Returns
    -------
    np.ndarray
        1D array of shape (n,) containing the DCT transfer coefficients.
    
    Raises
    ------
    ValueError
        If an unknown kernel_type is specified.
    """
    k = np.arange(n)
    
    if kernel_type == 'boxcar':
        # Analytical Boxcar: H = sin(W * pi * k / (2*n)) / (W * sin(pi * k / (2*n)))
        # Handle k=0 case separately to avoid division by zero
        theta_half = (np.pi * k) / (2 * n)
        numerator = np.sin(width * theta_half)
        denominator = np.sin(theta_half)
        H = np.zeros(n)
        H[0] = 1.0
        mask = k > 0
        
        # Avoid division by zero if denominator is 0 (should only be at k=0 which is handled)
        # But for safety in floating point:
        valid = np.abs(denominator) > 1e-10
        calc_mask = mask & valid
        
        if np.any(calc_mask):
             H[calc_mask] = numerator[calc_mask] / (denominator[calc_mask] * width)
             
        return H
        
    elif kernel_type == 'gaussian':
        # Sigma from width approximation (W ~ sqrt(12)*sigma for similar variance)
        sigma = width / np.sqrt(12)
        omega_k = (np.pi * k) / n
        return np.exp(-0.5 * (omega_k * sigma)**2)
        
    elif kernel_type == 'savgol':
        # Discrete implementation
        poly = kwargs.get('poly', 3)
        w_int = int(np.round(width))
        if w_int < poly + 2: w_int = poly + 2
        if w_int % 2 == 0: w_int += 1
        
        try:
            coeffs = scipy.signal.savgol_coeffs(w_int, poly)
        except ValueError:
            # Fallback if window is too small for poly
            return np.ones(n)

        # DCT Transfer Function for Symmetric FIR: c[0] + 2*sum(c[i]*cos(i*w_k))
        w_k = np.pi * k / n
        mid = (len(coeffs) - 1) // 2
        H = np.ones(n) * coeffs[mid]
        for i in range(1, mid + 1):
            H += 2 * coeffs[mid + i] * np.cos(i * w_k)
        return H
        
    elif kernel_type == 'hanning':
        # Discrete implementation
        w_int = int(np.round(width))
        if w_int < 3: w_int = 3
        if w_int % 2 == 0: w_int += 1
        
        window = scipy.signal.windows.hann(w_int)
        window /= np.sum(window)
        
        w_k = np.pi * k / n
        mid = (len(window) - 1) // 2
        H = np.ones(n) * window[mid]
        for i in range(1, mid + 1):
            H += 2 * window[mid + i] * np.cos(i * w_k)
        return H
    
    elif kernel_type == 'boxcar_discrete':
        # Discrete Boxcar
        # Odd integer width required for symmetric kernel
        w_int = int(np.round(width))
        if w_int < 1: 
            w_int = 1
        if w_int % 2 == 0:  
            w_int += 1
        
        # DCT Transfer Function for discrete boxcar
        # H[k] = (1 + 2*sum_{j=1}^{m} cos(j*w_k)) / w_int
        w_k = (np.pi * k) / n
        H = np.ones(n)
        half_width = (w_int - 1) // 2
        for j in range(1, half_width + 1):
            H += 2 * np.cos(j * w_k)
        H /= w_int
        return H
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def compute_polar_transfer_functions(
    data_shape: Tuple[int, int], 
    az_res_deg: float, 
    width_pixels: float, 
    kernel_type: str = 'boxcar', 
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes transfer functions for polar coordinate smoothing.
    
    The azimuth kernel width adapts with range to maintain a constant physical width.
    The range kernel width is fixed.
    
    Parameters
    ----------
    data_shape : tuple
        Shape of data array (n_az, n_range). Axis 0: Azimuth, Axis 1: Range.
    az_res_deg : float
        Angular resolution in degrees.
    width_pixels : float
        Target smoothing width in pixels (physical width reference).
    kernel_type : str
        Type of smoothing kernel.
    **kwargs : dict
        Additional parameters for specific kernels.
    
    Returns
    -------
    H_az : np.ndarray
        Azimuth transfer function of shape (n_az, n_range).
    H_range : np.ndarray
        Range transfer function of shape (n_range,).
    """
    n_az, n_range = data_shape
    
    # Azimuth resolution in radians
    az_res_rad = np.deg2rad(az_res_deg)
    
    # 1. Azimuth Transfer Function (varies with range)
    # We construct it as (n_range, n_az) first for clearer iteration over range
    H_az_T = np.zeros((n_range, n_az))
    
    for i in range(n_range):
        # Range index (1-based to avoid div by zero at strict origin if needed, 
        # though usually gate 0 is at some distance. Here we assume gate indices map effectively to distance ratios)
        # Using i+1 assumes the first bin is non-zero distance or small distance.
        r_idx = i + 1
        
        # Calculate effective beam width for this range index
        # physical_width = width_pixels * pixel_size
        # But we want constant physical width.
        # width_in_beams = physical_width / (r * az_current_res)
        # The logic: width_pixels is "nominal" pixels (e.g. at a ref distance or just a number).
        # Actually, if width_pixels is 5, it means "5 range-gate-sizes" physically? 
        # Or just a dimensionless smoothing parameter?
        # The original code used: w_beams = width_pixels / (r_idx * az_res_rad)
        # This implies width_pixels implies a physical arc length of (width_pixels * unit_range * unit_angle)?
        # Let's interpret width_pixels as "Width in units of Range Bins".
        # So we want an arc length equal to `width_pixels` * `range_bin_size`.
        # Arc length at range `r` for angle `theta` is `r * theta`.
        # So `w_physical = r * w_beams * az_res_rad`.
        # We want `w_physical = width_pixels * range_bin_size`? 
        # NOTE: This scaling assumes r_idx captures the geometry.
        
        w_beams = width_pixels / (r_idx * az_res_rad)
        
        # Don't let it blow up near origin or become too small
        # w_beams = max(w_beams, 0.1) 
        
        H_az_T[i, :] = get_transfer_function(n_az, kernel_type, w_beams, **kwargs)
    
    # Transpose to (n_az, n_range)
    H_az = H_az_T.T
    
    # 2. Range Transfer Function (fixed width in pixels)
    H_range = get_transfer_function(n_range, kernel_type, width_pixels, **kwargs)
    
    return H_az, H_range

def apply_polar_dct_smoothing(
    data: np.ndarray, 
    az_res_deg: float, 
    width_pixels: float, 
    kernel_type: str = 'boxcar', 
    **kwargs
) -> np.ndarray:
    """
    Apply DCT-based smoothing to polar coordinate data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_az, n_range). NaNs are supported.
    az_res_deg : float
        Angular resolution in degrees.
    width_pixels : float
        Smoothing width in pixels.
    kernel_type : str
        Type of smoothing kernel.
    **kwargs : dict
        Additional parameters for specific kernels.
    
    Returns
    -------
    np.ndarray
        Smoothed data array with the same shape as input.
    """
    # Create NaN mask to restore later if desired, or just to know where they were
    mask = np.isnan(data)
    
    # Fill NaNs
    # Interpolate along range (axis 1)
    data_filled = interpolate_2d_along_axis(data, axis=1)
    # Optionally interpolate along azimuth if needed, but range is usually sufficient for rays
    
    # Get transfer functions
    H_az, H_range = compute_polar_transfer_functions(
        data.shape, az_res_deg, width_pixels, kernel_type, **kwargs
    )
    
    # Apply Azimuth Smoothing (axis 0) via DCT
    # Type 2 DCT is standard for data extrapolation implication (even symmetry at boundaries)
    # Ortho normalization preserves energy scale
    X_az = scipy.fft.dct(data_filled, axis=0, type=2, norm='ortho')
    Y_az = X_az * H_az
    y_az = scipy.fft.idct(Y_az, axis=0, type=2, norm='ortho')
    
    # Apply Range Smoothing (axis 1) via DCT
    # H_range is 1D (n_range,), need to broadcast to (n_az, n_range)
    # Broadcasting happens automatically on last axis if shapes match
    X_range = scipy.fft.dct(y_az, axis=1, type=2, norm='ortho')
    Y_range = X_range * H_range  # Broadcasts (n_range,) across (n_az, n_range)
    y_result = scipy.fft.idct(Y_range, axis=1, type=2, norm='ortho')
    
    # Restore NaNs? 
    # Usually in smoothing we WANT to fill gaps, so we might not restore all NaNs.
    # But strictly, if a pixel was invalid, maybe we flag it. 
    # The original notebook restored them: `y_result[mask] = np.nan`
    # Let's keep that behavior for consistency with "smoothing" not "inpainting", 
    # although DCT does inpaint effectively.
    y_result[mask] = np.nan
    
    return y_result
