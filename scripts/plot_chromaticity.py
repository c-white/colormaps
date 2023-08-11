#! /usr/bin/env python3

"""
Script for making chromaticity diagrams.
"""

# Numerical modules
import numpy as np

# Plotting modules
import colorspacious as cs
import matplotlib
matplotlib.use('agg')
matplotlib.rc('font', size=8, family='serif')
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt

# Custom color modules
import matching_functions as mf
import standard_illuminants as si

# Main function
def main():

  # Parameters
  filename = 'plots/chromaticity.{0}.png'
  names = ('xy', 'viridis', 'inferno', 'magma', 'plasma')
  c_cgs = 2.99792458e10;
  h_cgs = 6.62607015e-27;
  kb_cgs = 1.380649e-16;

  # Plotting parameters - layout
  fig_width = 3.35
  aspect = 1.0
  lmar_frac = 0.14
  rmar_frac = 0.02
  bmar_frac = 0.11
  tmar_frac = 0.02
  dpi = 600

  # Plotting parameters - axes
  axes = {}
  axes['xy'] = {}
  axes['xy']['x_label'] = '$x$'
  axes['xy']['x_labelpad'] = -1.0
  axes['xy']['x_lim'] = (-0.05, 1.05)
  axes['xy']['x_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['xy']['y_label'] = '$y$'
  axes['xy']['y_labelpad'] = 3.0
  axes['xy']['y_lim'] = (-0.05, 1.05)
  axes['xy']['y_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['viridis'] = {}
  axes['viridis']['x_label'] = '$x$'
  axes['viridis']['x_labelpad'] = -1.0
  axes['viridis']['x_lim'] = (-0.05, 1.05)
  axes['viridis']['x_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['viridis']['y_label'] = '$y$'
  axes['viridis']['y_labelpad'] = 3.0
  axes['viridis']['y_lim'] = (-0.05, 1.05)
  axes['viridis']['y_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['inferno'] = {}
  axes['inferno']['x_label'] = '$x$'
  axes['inferno']['x_labelpad'] = -1.0
  axes['inferno']['x_lim'] = (-0.05, 1.05)
  axes['inferno']['x_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['inferno']['y_label'] = '$y$'
  axes['inferno']['y_labelpad'] = 3.0
  axes['inferno']['y_lim'] = (-0.05, 1.05)
  axes['inferno']['y_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['magma'] = {}
  axes['magma']['x_label'] = '$x$'
  axes['magma']['x_labelpad'] = -1.0
  axes['magma']['x_lim'] = (-0.05, 1.05)
  axes['magma']['x_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['magma']['y_label'] = '$y$'
  axes['magma']['y_labelpad'] = 3.0
  axes['magma']['y_lim'] = (-0.05, 1.05)
  axes['magma']['y_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['plasma'] = {}
  axes['plasma']['x_label'] = '$x$'
  axes['plasma']['x_labelpad'] = -1.0
  axes['plasma']['x_lim'] = (-0.05, 1.05)
  axes['plasma']['x_tick_locs'] = np.linspace(0.0, 1.0, 6)
  axes['plasma']['y_label'] = '$y$'
  axes['plasma']['y_labelpad'] = 3.0
  axes['plasma']['y_lim'] = (-0.05, 1.05)
  axes['plasma']['y_tick_locs'] = np.linspace(0.0, 1.0, 6)

  # Plotting parameters - shading
  shade_facecolor = (1.0/3.0, 1.0/3.0, 1.0/3.0)
  shade_edgecolor = 'none'

  # Plotting parameters - lines and points
  mono_linestyle = '-'
  mono_linewidth = 0.5
  mono_color = 'k'
  mono_marker = 'o'
  mono_markersize = 3.0
  mono_sample_color = 'k'
  mono_edgecolors = 'none'
  purple_linestyle = '-'
  purple_linewidth = 0.5
  purple_color = 'k'
  blackbody_linestyle = '-'
  blackbody_linewidth = 0.5
  blackbody_color = 'k'
  blackbody_marker = 'o'
  blackbody_markersize = 3.0
  blackbody_sample_color = 'k'
  blackbody_edgecolors = 'none'
  cmap_linestyle = '-'
  cmap_linewidth = 0.5
  cmap_color = 'k'
  ee_marker = 'o'
  ee_markersize = 3.0
  ee_color = 'k'
  ee_edgecolors = 'none'
  d65_marker = 'o'
  d65_markersize = 3.0
  d65_color = 'k'
  d65_edgecolors = 'none'
  rgb_linestyle = '-'
  rgb_linewidth = 0.5
  rgb_color = 'k'

  # Plotting parameters - gamut
  x_grid = np.linspace(0.0, 1.0, 513)
  y_grid = np.linspace(0.0, 1.0, 513)
  gamut_ee_yy = 1.0
  gamut_interpolation = 'none'
  gamut_origin = 'lower'
  gamut_extent = (0.0, 1.0, 0.0, 1.0)
  gamut_gray = 2.0/3.0

  # Calculate monochromatic locus
  if not np.all(mf.match_lambdas == np.linspace(390.0, 830.0, 441)):
    raise RuntimeError('Expecting different abscissas.')
  mono_lambdas = np.copy(mf.match_lambdas)[10:-130]
  mono_xx = np.copy(mf.match_x)[10:-130]
  mono_yy = np.copy(mf.match_y)[10:-130]
  mono_zz = np.copy(mf.match_z)[10:-130]
  mono_x = mono_xx / (mono_xx + mono_yy + mono_zz)
  mono_y = mono_yy / (mono_xx + mono_yy + mono_zz)
  mono_sample_inds = np.arange(301)[::25]

  # Calculate blackbody locus
  blackbody_tt = np.concatenate((np.linspace(1000.0, 30000.0, 291), [np.inf]))
  blackbody_xxyyzz = np.empty((len(blackbody_tt), 3))
  for n, tt in enumerate(blackbody_tt):
    lambdas_cgs = mf.match_lambdas * 1.0e-7
    if tt == np.inf:
      intensities = 1.0 / lambdas_cgs ** 4
    else:
      intensities = 1.0 / (lambdas_cgs ** 5 * np.expm1(h_cgs * c_cgs / (lambdas_cgs * kb_cgs * tt)))
    blackbody_xxyyzz[n,:] = mf.spectrum_to_xyz(intensities)
  blackbody_x = blackbody_xxyyzz[:,0] / np.sum(blackbody_xxyyzz, axis=1)
  blackbody_y = blackbody_xxyyzz[:,1] / np.sum(blackbody_xxyyzz, axis=1)
  blackbody_sample_inds = np.array((15, 40, 90, 190, 291))

  # Calculate colormap loci
  viridis_rgb = plt.get_cmap('viridis')(np.linspace(0.0, 1.0, 256))[:,:3]
  viridis_xyyy = cs.cspace_convert(viridis_rgb, 'sRGB255', 'xyY1')
  viridis_x = viridis_xyyy[:,0]
  viridis_y = viridis_xyyy[:,1]
  inferno_rgb = plt.get_cmap('inferno')(np.linspace(0.0, 1.0, 256))[:,:3]
  inferno_xyyy = cs.cspace_convert(inferno_rgb, 'sRGB255', 'xyY1')
  inferno_x = inferno_xyyy[:,0]
  inferno_y = inferno_xyyy[:,1]
  magma_rgb = plt.get_cmap('magma')(np.linspace(0.0, 1.0, 256))[:,:3]
  magma_xyyy = cs.cspace_convert(magma_rgb, 'sRGB255', 'xyY1')
  magma_x = magma_xyyy[:,0]
  magma_y = magma_xyyy[:,1]
  plasma_rgb = plt.get_cmap('plasma')(np.linspace(0.0, 1.0, 256))[:,:3]
  plasma_xyyy = cs.cspace_convert(plasma_rgb, 'sRGB255', 'xyY1')
  plasma_x = plasma_xyyy[:,0]
  plasma_y = plasma_xyyy[:,1]

  # Calculate illuminant E
  ee_x = 1.0/3.0
  ee_y = 1.0/3.0

  # Calculate illuminant D65
  if not np.all(si.lambdas_d65 == np.linspace(300.0, 780.0, 97)):
    raise RuntimeError('Expecting different abscissas.')
  d65_xxyyzz = mf.spectrum_to_xyz(np.interp(mf.match_lambdas, si.lambdas_d65, si.intensities_d65, left=0.0, right=0.0))
  d65_x = d65_xxyyzz[0] / np.sum(d65_xxyyzz)
  d65_y = d65_xxyyzz[1] / np.sum(d65_xxyyzz)

  # Calculate RGB triangle
  rgb_xxyyzz = mf.rgb_to_xyz(((255, 0, 0), (0, 255, 0), (0, 0, 255)))
  red_x = rgb_xxyyzz[0,0] / np.sum(rgb_xxyyzz[0,:])
  red_y = rgb_xxyyzz[0,1] / np.sum(rgb_xxyyzz[0,:])
  green_x = rgb_xxyyzz[1,0] / np.sum(rgb_xxyyzz[1,:])
  green_y = rgb_xxyyzz[1,1] / np.sum(rgb_xxyyzz[1,:])
  blue_x = rgb_xxyyzz[2,0] / np.sum(rgb_xxyyzz[2,:])
  blue_y = rgb_xxyyzz[2,1] / np.sum(rgb_xxyyzz[2,:])

  # Calculate RGB gamut
  x = 0.5 * (x_grid[:-1] + x_grid[1:])
  y = 0.5 * (y_grid[:-1] + y_grid[1:])
  gamut_x, gamut_y = np.meshgrid(x, y)
  gamut_red_yy = rgb_xxyyzz[0,1]
  gamut_green_yy = rgb_xxyyzz[1,1]
  gamut_blue_yy = rgb_xxyyzz[2,1]
  yellow_mask = triangle_mask(red_x, red_y, green_x, green_y, ee_x, ee_y, gamut_x, gamut_y)
  yellow_d = (red_x - green_x) * (ee_y - gamut_y) - (red_y - green_y) * (ee_x - gamut_x)
  yellow_x = ((ee_x - gamut_x) * (red_x * green_y - red_y * green_x) - (red_x - green_x) * (ee_x * gamut_y - ee_y * gamut_x)) / yellow_d
  yellow_y = ((ee_y - gamut_y) * (red_x * green_y - red_y * green_x) - (red_y - green_y) * (ee_x * gamut_y - ee_y * gamut_x)) / yellow_d
  yellow_f1 = np.hypot(yellow_x - red_x, yellow_y - red_y) / np.hypot(green_x - red_x, green_y - red_y)
  yellow_f2 = np.hypot(gamut_x - yellow_x, gamut_y - yellow_y) / np.hypot(ee_x - yellow_x, ee_y - yellow_y)
  yellow_yy = (1.0 - yellow_f2) * ((1.0 - yellow_f1) * gamut_red_yy + yellow_f1 * gamut_green_yy) + yellow_f2 * gamut_ee_yy
  cyan_mask = triangle_mask(green_x, green_y, blue_x, blue_y, ee_x, ee_y, gamut_x, gamut_y)
  cyan_d = (green_x - blue_x) * (ee_y - gamut_y) - (green_y - blue_y) * (ee_x - gamut_x)
  cyan_x = ((ee_x - gamut_x) * (green_x * blue_y - green_y * blue_x) - (green_x - blue_x) * (ee_x * gamut_y - ee_y * gamut_x)) / cyan_d
  cyan_y = ((ee_y - gamut_y) * (green_x * blue_y - green_y * blue_x) - (green_y - blue_y) * (ee_x * gamut_y - ee_y * gamut_x)) / cyan_d
  cyan_f1 = np.hypot(cyan_x - green_x, cyan_y - green_y) / np.hypot(blue_x - green_x, blue_y - green_y)
  cyan_f2 = np.hypot(gamut_x - cyan_x, gamut_y - cyan_y) / np.hypot(ee_x - cyan_x, ee_y - cyan_y)
  cyan_yy = (1.0 - cyan_f2) * ((1.0 - cyan_f1) * gamut_green_yy + cyan_f1 * gamut_blue_yy) + cyan_f2 * gamut_ee_yy
  magenta_mask = triangle_mask(blue_x, blue_y, red_x, red_y, ee_x, ee_y, gamut_x, gamut_y)
  magenta_d = (blue_x - red_x) * (ee_y - gamut_y) - (blue_y - red_y) * (ee_x - gamut_x)
  magenta_x = ((ee_x - gamut_x) * (blue_x * red_y - blue_y * red_x) - (blue_x - red_x) * (ee_x * gamut_y - ee_y * gamut_x)) / magenta_d
  magenta_y = ((ee_y - gamut_y) * (blue_x * red_y - blue_y * red_x) - (blue_y - red_y) * (ee_x * gamut_y - ee_y * gamut_x)) / magenta_d
  magenta_f1 = np.hypot(magenta_x - blue_x, magenta_y - blue_y) / np.hypot(red_x - blue_x, red_y - blue_y)
  magenta_f2 = np.hypot(gamut_x - magenta_x, gamut_y - magenta_y) / np.hypot(ee_x - magenta_x, ee_y - magenta_y)
  magenta_yy = (1.0 - magenta_f2) * ((1.0 - magenta_f1) * gamut_blue_yy + magenta_f1 * gamut_red_yy) + magenta_f2 * gamut_ee_yy
  gamut_yy = np.ones_like(gamut_y)
  gamut_yy = np.where(yellow_mask, yellow_yy, gamut_yy)
  gamut_yy = np.where(cyan_mask, cyan_yy, gamut_yy)
  gamut_yy = np.where(magenta_mask, magenta_yy, gamut_yy)
  gamut_xx = gamut_yy * gamut_x / gamut_y
  gamut_yy = gamut_yy * np.ones_like(gamut_y)
  gamut_zz = gamut_yy * (1.0 - gamut_x - gamut_y) / gamut_y
  gamut_xxyyzz = np.concatenate((gamut_xx[...,None], gamut_yy[...,None], gamut_zz[...,None]), axis=-1)
  gamut_rgb = mf.xyz_to_rgb(gamut_xxyyzz)

  # Calculate RGB gamut masking
  rgb_mask = triangle_mask(red_x, red_y, green_x, green_y, blue_x, blue_y, gamut_x, gamut_y)
  gamut_rgb = np.where(rgb_mask[...,None], gamut_rgb, gamut_gray)

  # Calculate layout
  panel_width = fig_width / (lmar_frac + 1.0 + rmar_frac)
  panel_height = panel_width / aspect
  lmar = lmar_frac * panel_width
  rmar = rmar_frac * panel_width
  bmar = bmar_frac * panel_width
  tmar = tmar_frac * panel_width
  fig_height = bmar + panel_height + tmar

  # Go through figures
  for name in names:

    # Prepare figure
    plt.figure(figsize=(fig_width,fig_height))
    ax = plt.subplot(1, 1, 1)

    # Shade inaccessible regions
    if name in ('xy', 'viridis', 'inferno', 'magma', 'plasma'):
      x = (axes[name]['x_lim'][0], 0.0)
      y1 = (axes[name]['y_lim'][0], axes[name]['y_lim'][0])
      y2 = (axes[name]['y_lim'][1], axes[name]['y_lim'][1])
      ax.fill_between(x, y1, y2, edgecolor=shade_edgecolor, facecolor=shade_facecolor)
      x = (axes[name]['x_lim'][0], axes[name]['x_lim'][1])
      y1 = (axes[name]['y_lim'][0], axes[name]['y_lim'][0])
      y2 = (0.0, 0.0)
      ax.fill_between(x, y1, y2, edgecolor=shade_edgecolor, facecolor=shade_facecolor)
      x = (axes[name]['x_lim'][0], axes[name]['x_lim'][1])
      y1 = (axes[name]['y_lim'][1], axes[name]['y_lim'][1])
      y2 = 1.0 - np.array(x)
      ax.fill_between(x, y1, y2, edgecolor=shade_edgecolor, facecolor=shade_facecolor)

    # Plot monochromatic locus
    if name in ('xy', 'viridis', 'inferno', 'magma', 'plasma'):
      ax.plot(mono_x, mono_y, linestyle=mono_linestyle, linewidth=mono_linewidth, color=mono_color)
    if name == 'xy':
      ax.scatter(mono_x[mono_sample_inds], mono_y[mono_sample_inds], s=mono_markersize, c=mono_sample_color, marker=mono_marker, edgecolors=mono_edgecolors)

    # Plot line of purples
    if name in ('xy', 'viridis', 'inferno', 'magma', 'plasma'):
      x = (mono_x[0], mono_x[-1])
      y = (mono_y[0], mono_y[-1])
      ax.plot(x, y, linestyle=purple_linestyle, linewidth=purple_linewidth, color=purple_color)

    # Plot blackbody locus
    if name == 'xy':
      ax.plot(blackbody_x, blackbody_y, linestyle=blackbody_linestyle, linewidth=blackbody_linewidth, color=blackbody_color)
      ax.scatter(blackbody_x[blackbody_sample_inds], blackbody_y[blackbody_sample_inds], s=blackbody_markersize, c=blackbody_sample_color, marker=blackbody_marker, edgecolors=blackbody_edgecolors)

    # Plot colormap loci
    if name == 'viridis':
      ax.plot(viridis_x, viridis_y, linestyle=cmap_linestyle, linewidth=cmap_linewidth, color=cmap_color)
    if name == 'inferno':
      ax.plot(inferno_x, inferno_y, linestyle=cmap_linestyle, linewidth=cmap_linewidth, color=cmap_color)
    if name == 'magma':
      ax.plot(magma_x, magma_y, linestyle=cmap_linestyle, linewidth=cmap_linewidth, color=cmap_color)
    if name == 'plasma':
      ax.plot(plasma_x, plasma_y, linestyle=cmap_linestyle, linewidth=cmap_linewidth, color=cmap_color)

    # Plot illuminant E
    if name == 'xy':
      ax.scatter(ee_x, ee_y, s=ee_markersize, c=ee_color, marker=ee_marker, edgecolors=ee_edgecolors)

    # Plot illuminant D65
    if name == 'xy':
      ax.scatter(d65_x, d65_y, s=d65_markersize, c=d65_color, marker=d65_marker, edgecolors=d65_edgecolors)

    # Plot RGB triangle
    if name in ('xy', 'viridis', 'inferno', 'magma', 'plasma'):
      x = (red_x, green_x, blue_x, red_x)
      y = (red_y, green_y, blue_y, red_y)
      ax.plot(x, y, linestyle=rgb_linestyle, linewidth=rgb_linewidth, color=rgb_color)

    # Plot RGB gamut
    if name in ('xy', 'viridis', 'inferno', 'magma', 'plasma'):
      ax.imshow(gamut_rgb, interpolation=gamut_interpolation, origin=gamut_origin, extent=gamut_extent)

    # Adjust axes
    ax.set_xlim(axes[name]['x_lim'])
    ax.set_xticks(axes[name]['x_tick_locs'])
    ax.set_xlabel(axes[name]['x_label'], labelpad=axes[name]['x_labelpad'])
    ax.set_ylim(axes[name]['y_lim'])
    ax.set_yticks(axes[name]['y_tick_locs'])
    ax.set_ylabel(axes[name]['y_label'], labelpad=axes[name]['y_labelpad'])

    # Adjust layout
    width = panel_width / fig_width
    height = panel_height / fig_height
    x0 = lmar / fig_width
    y0 = bmar / fig_height
    ax.set_position((x0, y0, width, height))

    # Save figure
    plt.savefig(filename.format(name), dpi=dpi)
    plt.close()

# Function for masking a triangle
def triangle_mask(a_x, a_y, b_x, b_y, c_x, c_y, p_x, p_y):
  ax = c_x - a_x
  ay = c_y - a_y
  bx = b_x - a_x
  by = b_y - a_y
  cx = p_x - a_x
  cy = p_y - a_y
  aa = ax * ax + ay * ay
  ab = ax * bx + ay * by
  ac = ax * cx + ay * cy
  bb = bx * bx + by * by
  bc = bx * cx + by * cy
  w = aa * bb - ab * ab
  u = bb * ac - ab * bc
  v = aa * bc - ab * ac
  mask = (u >= 0) & (v >= 0) & (u + v < w)
  return mask

# Execute main function
if __name__ == '__main__':
  main()
