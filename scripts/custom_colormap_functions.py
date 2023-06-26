"""
Functions for defining custom colormaps.
"""

# Python standard modules
import warnings

# Numerical modules
import numpy as np
from scipy.optimize import least_squares

# Color modules
import colorspacious as cs
import matplotlib.cm as cm
import matplotlib.colors as colors

# Segmented colormap that is piecewise linear in J'a'b'
def linear_segment(anchors, samples_below_jjab, samples_above_jjab, name='linear_segment', min_points=1024, tol=1.0e-6):

  # Prepare abscissas
  x = np.linspace(0.0, 1.0, min_points)
  for anchor in anchors:
    if np.min(np.abs(x - anchor) > tol):
      ind = np.where(x > anchor)[0][0]
      x = np.concatenate((x[:ind], [anchor], x[ind:]))
  num_points = len(x)

  # Interpolate samples in perceptually uniform space
  below_jjab = np.empty((num_points, 3))
  above_jjab = np.empty((num_points, 3))
  for n in range(num_points):
    if x[n] < anchors[0]:
      val_jjab = samples_below_jjab[0,:]
      below_jjab[n,:] = val_jjab
      above_jjab[n,:] = val_jjab
    elif x[n] > anchors[-1]:
      val_jjab = samples_above_jjab[-1,:]
      below_jjab[n,:] = val_jjab
      above_jjab[n,:] = val_jjab
    elif x[n] == anchors[-1]:
      below_jjab[n,:] = samples_below_jjab[-1,:]
      above_jjab[n,:] = samples_above_jjab[-1,:]
    else:
      ind = np.where(anchors <= x[n])[0][-1]
      if x[n] == anchors[ind]:
        below_jjab[n,:] = samples_below_jjab[ind,:]
        above_jjab[n,:] = samples_above_jjab[ind,:]
      else:
        frac = (x[n] - anchors[ind]) / (anchors[ind+1] - anchors[ind])
        val_jjab = (1.0 - frac) * samples_above_jjab[ind,:] + frac * samples_below_jjab[ind+1,:]
        below_jjab[n,:] = val_jjab
        above_jjab[n,:] = val_jjab

  # Convert values to RGB
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)
    below_rgb1 = cs.cspace_convert(below_jjab, cs.CAM02UCS, 'sRGB1')
    above_rgb1 = cs.cspace_convert(above_jjab, cs.CAM02UCS, 'sRGB1')

  # Clip colors
  below_rgb1 = np.clip(below_rgb1, 0.0, 1.0)
  above_rgb1 = np.clip(above_rgb1, 0.0, 1.0)

  # Create colormap
  segment_data = {}
  segment_data['red'] = np.hstack((x[:,None], below_rgb1[:,:1], above_rgb1[:,:1]))
  segment_data['green'] = np.hstack((x[:,None], below_rgb1[:,1:2], above_rgb1[:,1:2]))
  segment_data['blue'] = np.hstack((x[:,None], below_rgb1[:,2:], above_rgb1[:,2:]))
  cmap = colors.LinearSegmentedColormap(name, segment_data, N=num_points)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Perceptually uniform helix
def helix_uniform(start_jjmmh, end_jjmmh, winding_num, name='helix_uniform', num_points=1024):

  # Parameters
  c1 = 0.007
  c2 = 0.0228
  hr_max_dev = np.pi/4.0
  end_frac = 0.05

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
    warnings.filterwarnings('ignore', 'divide by zero encountered in divide', RuntimeWarning)
    rgb1 = cs.cspace_convert(jjab, cs.CAM02UCS, 'sRGB1')

  # Clip colors
  rgb1 = np.clip(rgb1, 0.0, 1.0)
  jjab = cs.cspace_convert(rgb1, 'sRGB1', cs.CAM02UCS)

  # Ensure clipped lightness is monotonic near ends
  ind_max = int(end_frac * (num_points - 1)) + 1
  if jjp[-1] - jjp[0] > 0.0 and np.any(np.diff(jjab[:ind_max,0]) < 0.0):
    ind = np.where(np.diff(jjab[:ind_max,0]) < 0.0)[0][-1] + 2
    fracs = (jjp[1:ind] - jjp[0]) / (jjp[ind] - jjp[0])
    rgb1[1:ind,:] = (1.0 - fracs[:,None]) * rgb1[:1,:] + fracs[:,None] * rgb1[ind:ind+1,:]
  elif jjp[-1] - jjp[0] < 0.0 and np.any(np.diff(jjab[:ind_max,0]) > 0.0):
    ind = np.where(np.diff(jjab[:ind_max,0]) > 0.0)[0][-1] + 2
    fracs = (jjp[1:ind] - jjp[0]) / (jjp[ind] - jjp[0])
    rgb1[1:ind,:] = (1.0 - fracs[:,None]) * rgb1[:1,:] + fracs[:,None] * rgb1[ind:ind+1,:]
  ind_min = int((1.0 - end_frac) * (num_points - 1))
  if jjp[-1] - jjp[0] > 0.0 and np.any(np.diff(jjab[ind_min:,0]) < 0.0):
    ind = np.where(np.diff(jjab[ind_min:,0]) < 0.0)[0][0] + ind_min - 1
    fracs = (jjp[ind+1:-1] - jjp[ind]) / (jjp[-1] - jjp[ind])
    rgb1[ind+1:-1,:] = (1.0 - fracs[:,None]) * rgb1[ind:ind+1,:] + fracs[:,None] * rgb1[-1:,:]
  elif jjp[-1] - jjp[0] < 0.0 and np.any(np.diff(jjab[ind_min:,0]) > 0.0):
    ind = np.where(np.diff(jjab[ind_min:,0]) > 0.0)[0][0] + ind_min - 1
    fracs = (jjp[ind+1:-1] - jjp[ind]) / (jjp[-1] - jjp[ind])
    rgb1[ind+1:-1,:] = (1.0 - fracs[:,None]) * rgb1[ind:ind+1,:] + fracs[:,None] * rgb1[-1:,:]

  # Create colormap
  x_rgb1 = [(x_val, rgb1_val) for x_val, rgb1_val in zip(x, rgb1)]
  cmap = colors.LinearSegmentedColormap.from_list(name, x_rgb1, N=num_points)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Utility bump function
def bump_function_aux(x, k):
  if x <= 0.0:
    return 1.0
  elif x >= 1.0:
    return 0.0
  else:
    return np.exp(x ** k / (x ** k - 1.0))
bump_function = np.vectorize(bump_function_aux)
