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

# Perceptually uniform grayscale colormap
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
  cmap = colors.LinearSegmentedColormap.from_list(name, x_rgb1, N=num_points)
  cm.register_cmap(name=name, cmap=cmap)
  return cmap

# Blue-pink perceptually uniform monotonic colormap
def cool_uniform(name='cool_uniform', **kwargs):

  # Parameters
  start_jjmmh = (10.0, 30.0, 230.0)
  end_jjmmh = (80.0, 50.0, 330.0)
  winding_num = 0

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Red-yellow perceptually uniform monotonic colormap
def warm_uniform(name='warm_uniform', **kwargs):

  # Parameters
  start_jjmmh = (20.0, 60.0, 0.0)
  end_jjmmh = (90.0, 100.0, 100.0)
  winding_num = 0

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap matching endpoints of viridis
def viridis_alt(name='viridis_alt', **kwargs):

  # Parameters
  cmap_base_name = 'viridis'
  start_val = 0.0
  end_val = 1.0
  winding_num = 0

  # Calculate endpoints
  start_rgb1 = cm.get_cmap(cmap_base_name)(start_val)[:-1]
  end_rgb1 = cm.get_cmap(cmap_base_name)(end_val)[:-1]
  start_jjmmh = cs.cspace_convert(start_rgb1, 'sRGB1', 'JMh')
  end_jjmmh = cs.cspace_convert(end_rgb1, 'sRGB1', 'JMh')

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap matching endpoints of plasma
def plasma_alt(name='plasma_alt', **kwargs):

  # Parameters
  cmap_base_name = 'plasma'
  start_val = 0.0
  end_val = 1.0
  winding_num = 1

  # Calculate endpoints
  start_rgb1 = cm.get_cmap(cmap_base_name)(start_val)[:-1]
  end_rgb1 = cm.get_cmap(cmap_base_name)(end_val)[:-1]
  start_jjmmh = cs.cspace_convert(start_rgb1, 'sRGB1', 'JMh')
  end_jjmmh = cs.cspace_convert(end_rgb1, 'sRGB1', 'JMh')

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap based on inferno
def inferno_alt(name='inferno_alt', **kwargs):

  # Parameters
  cmap_base_name = 'inferno'
  start_val = 0.1
  end_val = 0.95
  start_factor = 2.0
  end_factor = 1.5
  winding_num = 1

  # Calculate endpoints
  start_rgb1 = cm.get_cmap(cmap_base_name)(start_val)[:-1]
  end_rgb1 = cm.get_cmap(cmap_base_name)(end_val)[:-1]
  start_jjmmh = cs.cspace_convert(start_rgb1, 'sRGB1', 'JMh')
  end_jjmmh = cs.cspace_convert(end_rgb1, 'sRGB1', 'JMh')
  start_jjmmh[1] *= start_factor
  end_jjmmh[1] *= end_factor

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap based on inferno and including black
def inferno_k(name='inferno_k', **kwargs):

  # Parameters
  cmap_base_name = 'inferno'
  a_val = 0.1
  b_val = 0.95
  a_mm_factor = 1.2
  b_mm_factor = 4.0
  a_loc = 0.1
  b_loc = 1.0
  start_jj_adjust = 0.0
  end_jj_adjust = None
  winding_num = 1

  # Calculate anchor points
  a_rgb1 = cm.get_cmap(cmap_base_name)(a_val)[:-1]
  b_rgb1 = cm.get_cmap(cmap_base_name)(b_val)[:-1]
  a_jjmmh = cs.cspace_convert(a_rgb1, 'sRGB1', 'JMh')
  b_jjmmh = cs.cspace_convert(b_rgb1, 'sRGB1', 'JMh')
  a_jjmmh[1] *= a_mm_factor
  b_jjmmh[1] *= b_mm_factor

  # Calculate endpoints
  start_jj = start_jj_adjust if start_jj_adjust is not None else a_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_jj = end_jj_adjust if end_jj_adjust is not None else b_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_mm = a_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_mm = b_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_h = a_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_h = b_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_jjmmh = (start_jj, start_mm, start_h)
  end_jjmmh = (end_jj, end_mm, end_h)

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap based on inferno and including white
def inferno_w(name='inferno_w', **kwargs):

  # Parameters
  cmap_base_name = 'inferno'
  a_val = 0.3
  b_val = 0.95
  a_mm_factor = 1.0
  b_mm_factor = 1.25
  a_loc = 0.0
  b_loc = 0.85
  start_jj_adjust = 10.0
  end_jj_adjust = 150.0
  winding_num = 1

  # Calculate anchor points
  a_rgb1 = cm.get_cmap(cmap_base_name)(a_val)[:-1]
  b_rgb1 = cm.get_cmap(cmap_base_name)(b_val)[:-1]
  a_jjmmh = cs.cspace_convert(a_rgb1, 'sRGB1', 'JMh')
  b_jjmmh = cs.cspace_convert(b_rgb1, 'sRGB1', 'JMh')
  a_jjmmh[1] *= a_mm_factor
  b_jjmmh[1] *= b_mm_factor

  # Calculate endpoints
  start_jj = start_jj_adjust if start_jj_adjust is not None else a_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_jj = end_jj_adjust if end_jj_adjust is not None else b_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_mm = a_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_mm = b_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_h = a_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_h = b_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_jjmmh = (start_jj, start_mm, start_h)
  end_jjmmh = (end_jj, end_mm, end_h)

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Perceptually uniform monotonic colormap based on inferno and including black and white
def inferno_kw(name='inferno_kw', **kwargs):

  # Parameters
  cmap_base_name = 'inferno'
  a_val = 0.1
  b_val = 1.0
  a_mm_factor = 0.9
  b_mm_factor = 2.0
  a_loc = 0.1
  b_loc = 0.9
  start_jj_adjust = 0.0
  end_jj_adjust = 150.0
  winding_num = 1

  # Calculate anchor points
  a_rgb1 = cm.get_cmap(cmap_base_name)(a_val)[:-1]
  b_rgb1 = cm.get_cmap(cmap_base_name)(b_val)[:-1]
  a_jjmmh = cs.cspace_convert(a_rgb1, 'sRGB1', 'JMh')
  b_jjmmh = cs.cspace_convert(b_rgb1, 'sRGB1', 'JMh')
  a_jjmmh[1] *= a_mm_factor
  b_jjmmh[1] *= b_mm_factor

  # Calculate endpoints
  start_jj = start_jj_adjust if start_jj_adjust is not None else a_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_jj = end_jj_adjust if end_jj_adjust is not None else b_jjmmh[0] + (b_jjmmh[0] - a_jjmmh[0]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_mm = a_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_mm = b_jjmmh[1] + (b_jjmmh[1] - a_jjmmh[1]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_h = a_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (0.0 - a_loc)
  end_h = b_jjmmh[2] + (b_jjmmh[2] + 360.0 * winding_num - a_jjmmh[2]) / (b_loc - a_loc) * (1.0 - b_loc)
  start_jjmmh = (start_jj, start_mm, start_h)
  end_jjmmh = (end_jj, end_mm, end_h)

  # Create colormap
  return custom_colormap_functions.helix_uniform(start_jjmmh, end_jjmmh, winding_num, name=name, **kwargs)

# Red-white-blue uniform lightness diverging colormap
def red_white_blue(name='red_white_blue', **kwargs):

  # Parameters
  red_arg = 0.1
  blue_arg = 0.9
  red_loc = 0.2
  blue_loc = 0.8
  white_jjp = 95.0
  white_left_ab = (0.0, 0.0)
  white_right_ab = (0.0, 0.0)
  white_ab_factor = 0.1
  color_stretch = 10.0
  balance_ab = True
  match_lightness = True
  num_half_intervals = 32

  # Define initial anchor values
  red_rgb1 = cm.get_cmap('RdBu')(red_arg)[:-1]
  blue_rgb1 = cm.get_cmap('RdBu')(blue_arg)[:-1]

  # Convert to perceptually uniform space
  red_jjab = cs.cspace_convert(red_rgb1, 'sRGB1', cs.CAM02UCS)
  blue_jjab = cs.cspace_convert(blue_rgb1, 'sRGB1', cs.CAM02UCS)
  if balance_ab:
    ap = 0.5 * (red_jjab[1] - blue_jjab[2])
    bp = 0.5 * (red_jjab[2] - blue_jjab[1])
    red_jjab[1:] = (ap, bp)
    blue_jjab[1:] = (-bp, -ap)

  # Adjust anchors
  red_jjab[0] = white_jjp + (red_jjab[0] - white_jjp) / (red_loc - 0.5) * (0.0 - 0.5)
  blue_jjab[0] = white_jjp + (blue_jjab[0] - white_jjp) / (blue_loc - 0.5) * (1.0 - 0.5)
  if match_lightness:
    end_jjp = 0.5 * (red_jjab[0] + blue_jjab[0])
    red_jjab[0] = end_jjp
    blue_jjab[0] = end_jjp
  white_left_ab = (1.0 - white_ab_factor) * np.array(white_left_ab) + white_ab_factor * red_jjab[1:]
  white_right_ab = (1.0 - white_ab_factor) * np.array(white_right_ab) + white_ab_factor * blue_jjab[1:]

  # Calculate anchor points and values for half intervals
  anchors_half = np.linspace(0.0, 1.0, num_half_intervals + 1)
  interp_half = custom_colormap_functions.bump_function(anchors_half, color_stretch)
  left_jjp = np.linspace(red_jjab[0], white_jjp, num_half_intervals + 1)
  left_ap = white_left_ab[0] + (red_jjab[1] - white_left_ab[0]) * interp_half
  left_bp = white_left_ab[1] + (red_jjab[2] - white_left_ab[1]) * interp_half
  right_jjp = np.linspace(blue_jjab[0], white_jjp, num_half_intervals + 1)
  right_ap = white_right_ab[0] + (blue_jjab[1] - white_right_ab[0]) * interp_half
  right_bp = white_right_ab[1] + (blue_jjab[2] - white_right_ab[1]) * interp_half

  # Assemble anchor points and values for full interval
  anchors = np.linspace(0.0, 1.0, 2 * num_half_intervals + 1)
  jjp = np.concatenate((left_jjp, right_jjp[-2::-1]))
  below_ap = np.concatenate((left_ap, right_ap[-2::-1]))
  below_bp = np.concatenate((left_bp, right_bp[-2::-1]))
  above_ap = np.concatenate((left_ap[:-1], right_ap[::-1]))
  above_bp = np.concatenate((left_bp[:-1], right_bp[::-1]))
  below_jjab = np.hstack((jjp[:,None], below_ap[:,None], below_bp[:,None]))
  above_jjab = np.hstack((jjp[:,None], above_ap[:,None], above_bp[:,None]))

  # Create colormap
  return custom_colormap_functions.linear_segment(anchors, below_jjab, above_jjab, name=name, **kwargs)

# Red-black-blue uniform lightness diverging colormap
def red_black_blue(name='red_black_blue', **kwargs):

  # Parameters
  red_arg = 0.2
  blue_arg = 0.8
  red_loc = 0.1
  blue_loc = 0.9
  black_jjp = 5.0
  black_left_ab = (0.0, 0.0)
  black_right_ab = (0.0, 0.0)
  black_ab_factor = 0.5
  color_stretch = 5.0
  balance_ab = True
  match_lightness = True
  num_half_intervals = 32

  # Define initial anchor values
  red_rgb1 = cm.get_cmap('RdBu')(red_arg)[:-1]
  blue_rgb1 = cm.get_cmap('RdBu')(blue_arg)[:-1]

  # Convert to perceptually uniform space
  red_jjab = cs.cspace_convert(red_rgb1, 'sRGB1', cs.CAM02UCS)
  blue_jjab = cs.cspace_convert(blue_rgb1, 'sRGB1', cs.CAM02UCS)
  if balance_ab:
    ap = 0.5 * (red_jjab[1] - blue_jjab[2])
    bp = 0.5 * (red_jjab[2] - blue_jjab[1])
    red_jjab[1:] = (ap, bp)
    blue_jjab[1:] = (-bp, -ap)

  # Adjust anchors
  red_jjab[0] = black_jjp + (red_jjab[0] - black_jjp) / (red_loc - 0.5) * (0.0 - 0.5)
  blue_jjab[0] = black_jjp + (blue_jjab[0] - black_jjp) / (blue_loc - 0.5) * (1.0 - 0.5)
  if match_lightness:
    end_jjp = 0.5 * (red_jjab[0] + blue_jjab[0])
    red_jjab[0] = end_jjp
    blue_jjab[0] = end_jjp
  black_left_ab = (1.0 - black_ab_factor) * np.array(black_left_ab) + black_ab_factor * red_jjab[1:]
  black_right_ab = (1.0 - black_ab_factor) * np.array(black_right_ab) + black_ab_factor * blue_jjab[1:]

  # Calculate anchor points and values for half intervals
  anchors_half = np.linspace(0.0, 1.0, num_half_intervals + 1)
  interp_half = custom_colormap_functions.bump_function(anchors_half, color_stretch)
  left_jjp = np.linspace(red_jjab[0], black_jjp, num_half_intervals + 1)
  left_ap = black_left_ab[0] + (red_jjab[1] - black_left_ab[0]) * interp_half
  left_bp = black_left_ab[1] + (red_jjab[2] - black_left_ab[1]) * interp_half
  right_jjp = np.linspace(blue_jjab[0], black_jjp, num_half_intervals + 1)
  right_ap = black_right_ab[0] + (blue_jjab[1] - black_right_ab[0]) * interp_half
  right_bp = black_right_ab[1] + (blue_jjab[2] - black_right_ab[1]) * interp_half

  # Assemble anchor points and values for full interval
  anchors = np.linspace(0.0, 1.0, 2 * num_half_intervals + 1)
  jjp = np.concatenate((left_jjp, right_jjp[-2::-1]))
  below_ap = np.concatenate((left_ap, right_ap[-2::-1]))
  below_bp = np.concatenate((left_bp, right_bp[-2::-1]))
  above_ap = np.concatenate((left_ap[:-1], right_ap[::-1]))
  above_bp = np.concatenate((left_bp[:-1], right_bp[::-1]))
  below_jjab = np.hstack((jjp[:,None], below_ap[:,None], below_bp[:,None]))
  above_jjab = np.hstack((jjp[:,None], above_ap[:,None], above_bp[:,None]))

  # Create colormap
  return custom_colormap_functions.linear_segment(anchors, below_jjab, above_jjab, name=name, **kwargs)
