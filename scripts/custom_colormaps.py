"""
Definitions of custom colormaps.
"""

# Python standard modules
import warnings

# Numerical modules
import numpy as np

# Color modules
import colorspacious as cs
import matplotlib.cm as cm
import matplotlib.colors as colors

# Custom color modules
import custom_colormap_functions

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

# Red-black-blue perceptually uniform diverging colormap
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
  return custom_colormap_functions.segment_uniform((0.0, 0.5, 1.0), (red_rgb1, black_rgb1, blue_rgb1), name=name, **kwargs)

# Blue-pink perceptually uniform monotonic colormap
def cool_uniform(name='cool_uniform', **kwargs):

  # Parameters
  start_jjmmh = (10.0, 30.0, 230.0)
  end_jjmmh = (80.0, 50.0, 330.0)
  winding_num = 0

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)
