"""
Definitions of custom colormaps.
"""

# Python standard modules
import warnings

# Modules
import colorspacious as cs
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import least_squares

# Perceptually uniform grayscale
def gray_uniform(name='gray_uniform', num_points=1024, colorspace='CAM02UCS'):

  # Prepare abscissas
  x = np.linspace(0.0, 1.0, num_points)

  # Calculate RGB colors
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', RuntimeWarning)
    if colorspace in ('CAM02UCS', 'CAM02LCD', 'CAM02SCD'):
      jjp = 100.0 * x
      ap = np.zeros_like(x)
      bp = np.zeros_like(x)
      jjab = np.hstack((jjp[:,None], ap[:,None], bp[:,None]))
      if colorspace == 'CAM02UCS':
        rgb1 = cs.cspace_convert(jjab, cs.CAM02UCS, 'sRGB1')
      if colorspace == 'CAM02LCS':
        rgb1 = cs.cspace_convert(jjab, cs.CAM02LCS, 'sRGB1')
      if colorspace == 'CAM02SCD':
        rgb1 = cs.cspace_convert(jjab, cs.CAM02SCD, 'sRGB1')
    elif colorspace == 'CIECAM02':
      jj = 100.0 * x
      cc = np.zeros_like(x)
      h = np.zeros_like(x)
      jjcch = np.hstack((jj[:,None], cc[:,None], h[:,None]))
      rgb1 = cs.cspace_convert(jjcch, 'JCh', 'sRGB1')
    elif colorspace == 'CIELab':
      ll = 100.0 * x
      a = np.zeros_like(x)
      b = np.zeros_like(x)
      llab = np.hstack((ll[:,None], a[:,None], b[:,None]))
      rgb1 = cs.cspace_convert(llab, 'CIELab', 'sRGB1')
    else:
      raise RuntimeError('Invalid colorspace')

  # Clip colors
  rgb1 = np.clip(rgb1, 0.0, 1.0)

  # Create colormap
  x_rgb1 = [(x_val, rgb1_val) for x_val, rgb1_val in zip(x, rgb1)]
  cmap = colors.LinearSegmentedColormap.from_list(name, x_rgb1)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Linear segment
def linear_segment(anchors, samples_rgb1, name='linear_segment', num_points=1024):

  # Prepare abscissas
  x = np.linspace(0.0, 1.0, num_points)

  # Convert samples to perceptually uniform space
  samples_jjab = cs.cspace_convert(samples_rgb1, 'sRGB1', cs.CAM02UCS)

  # Interpolate samples in perceptually uniform space
  jjp = np.interp(x, anchors, samples_jjab[:,0])
  ap = np.interp(x, anchors, samples_jjab[:,1])
  bp = np.interp(x, anchors, samples_jjab[:,2])
  jjab = np.hstack((jjp[:,None], ap[:,None], bp[:,None]))

  # Calculate RGB values
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide', RuntimeWarning)
    rgb1 = cs.cspace_convert(jjab, cs.CAM02UCS, 'sRGB1')

  # Clip colors
  rgb1 = np.clip(rgb1, 0.0, 1.0)

  # Create colormap
  x_rgb1 = [(x_val, rgb1_val) for x_val, rgb1_val in zip(x, rgb1)]
  cmap = colors.LinearSegmentedColormap.from_list(name, x_rgb1)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Perceptually uniform helix
def helix_uniform(start_jjmmh, end_jjmmh, winding_num, name='helix_uniform', num_points=1024):

  # Parameters
  c1 = 0.007
  c2 = 0.0228
  hr_max_dev = np.pi / 4.0

  # Prepare abscissas
  x = np.linspace(0.0, 1.0, num_points)

  # Calculate endpoints in perceptually uniform space
  jj_start = start_jjmmh[0]
  jjp_start = (1.0 + 100.0 * c1) * jj_start / (1.0 + c1 * jj_start)
  mm_start = start_jjmmh[1]
  mmp_start = np.log(1.0 + c2 * mm_start) / c2
  h_start = start_jjmmh[2]
  hr_start = h_start * np.pi / 180.0
  jj_end = end_jjmmh[0]
  jjp_end = (1.0 + 100.0 * c1) * jj_end / (1.0 + c1 * jj_end)
  mm_end = end_jjmmh[1]
  mmp_end = np.log(1.0 + c2 * mm_end) / c2
  h_end = end_jjmmh[2]
  hr_end = h_end * np.pi / 180.0

  # Calculate regular helix
  jjp = np.linspace(jjp_start, jjp_end, num_points)
  mmp = np.linspace(mmp_start, mmp_end, num_points)
  hr = np.linspace(hr_start, hr_end + 2.0 * np.pi * winding_num, num_points)

  # Make helix uniform in color difference
  def res(hr_int_vals):
    hr_vals = np.concatenate(([hr[0]], hr_int_vals, [hr[-1]]))
    ap_vals = mmp * np.cos(hr_vals)
    bp_vals = mmp * np.sin(hr_vals)
    deep_vals = np.hypot(np.diff(jjp), np.hypot(np.diff(ap_vals), np.diff(bp_vals)))
    return deep_vals - np.mean(deep_vals)
  def jac(hr_int_vals):
    hr_vals = np.concatenate(([hr[0]], hr_int_vals, [hr[-1]]))
    ap_vals = mmp * np.cos(hr_vals)
    bp_vals = mmp * np.sin(hr_vals)
    deep_vals = np.hypot(np.diff(jjp), np.hypot(np.diff(ap_vals), np.diff(bp_vals)))
    jacobian = np.zeros((num_points - 1, num_points - 2))
    jacobian[0,0] = mmp[0] * mmp[1] / deep_vals[0] * np.sin(hr_vals[1] - hr_vals[0])
    for k in range(1, num_points - 2):
      jacobian[k,k-1] = mmp[k] * mmp[k+1] / deep_vals[k] * np.sin(hr_vals[k] - hr_vals[k+1])
      jacobian[k,k] = -jacobian[k,k-1]
    jacobian[-1,-1] = mmp[-2] * mmp[-1] / deep_vals[-1] * np.sin(hr_vals[-1] - hr_vals[-2])
    return jacobian
  jac_sparsity = np.zeros((num_points - 1, num_points - 2))
  jac_sparsity[0,0] = 1.0
  for k in range(1, num_points - 2):
    jac_sparsity[k,k-1:k+1] = 1.0
  jac_sparsity[-1,-1] = 1.0
  bounds = (hr[1:-1] - hr_max_dev, hr[1:-1] + hr_max_dev)
  sol = least_squares(res, hr[1:-1], jac=jac, bounds=bounds, jac_sparsity=jac_sparsity)
  hr = np.concatenate(([hr[0]], sol.x, [hr[-1]]))

  # Calculate RGB values
  ap = mmp * np.cos(hr)
  bp = mmp * np.sin(hr)
  jjab = np.hstack((jjp[:,None], ap[:,None], bp[:,None]))
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide', RuntimeWarning)
    rgb1 = cs.cspace_convert(jjab, cs.CAM02UCS, 'sRGB1')

  # Clip colors
  rgb1 = np.clip(rgb1, 0.0, 1.0)

  # Create colormap
  x_rgb1 = [(x_val, rgb1_val) for x_val, rgb1_val in zip(x, rgb1)]
  cmap = colors.LinearSegmentedColormap.from_list(name, x_rgb1)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Linear segment instance
def red_black_blue(name='red_black_blue', **kwargs):

  # Parameters
  red_arg = 0.15
  blue_arg = 0.75
  red_loc = 0.1
  blue_loc = 1.0
  black_val = 0.0

  # Define initial anchor values
  red_rgb1 = cm.get_cmap('RdBu')(red_arg)[:-1]
  blue_rgb1 = cm.get_cmap('RdBu')(blue_arg)[:-1]
  black_rgb1 = (black_val, black_val, black_val)

  # Convert to perceptually uniform space
  red_jjab = cs.cspace_convert(red_rgb1, 'sRGB1', cs.CAM02UCS)
  blue_jjab = cs.cspace_convert(blue_rgb1, 'sRGB1', cs.CAM02UCS)
  black_jjab = cs.cspace_convert(black_rgb1, 'sRGB1', cs.CAM02UCS)

  # Adjust anchors
  red_jjab[0] = black_jjab[0] + (red_jjab[0] - black_jjab[0]) / (red_loc - 0.5) * (0.0 - 0.5)
  blue_jjab[0] = black_jjab[0] + (blue_jjab[0] - black_jjab[0]) / (blue_loc - 0.5) * (1.0 - 0.5)
  jjp = 0.5 * (red_jjab[0] + blue_jjab[0])
  red_jjab[0] = jjp
  blue_jjab[0] = jjp
  black_jjab[1] = 0.0
  black_jjab[2] = 0.0

  # Convert to RGB
  red_rgb1 = cs.cspace_convert(red_jjab, cs.CAM02UCS, 'sRGB1')
  blue_rgb1 = cs.cspace_convert(blue_jjab, cs.CAM02UCS, 'sRGB1')
  black_rgb1 = cs.cspace_convert(black_jjab, cs.CAM02UCS, 'sRGB1')

  # Clip anchors
  red_rgb1 = np.clip(red_rgb1, 0.0, 1.0)
  blue_rgb1 = np.clip(blue_rgb1, 0.0, 1.0)
  black_rgb1 = np.clip(black_rgb1, 0.0, 1.0)

  # Create colormap
  return linear_segment((0.0, 0.5, 1.0), (red_rgb1, black_rgb1, blue_rgb1), name=name, **kwargs)

# Perceptually uniform helix instance
def cool_uniform(name='cool_uniform', **kwargs):
  return helix_uniform((10.0, 30.0, 230.0), (80.0, 50.0, 330.0), 0, name=name, **kwargs)
