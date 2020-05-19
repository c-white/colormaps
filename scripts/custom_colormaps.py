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
