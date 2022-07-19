#! /usr/bin/env python

"""
Script for plotting all images with all colormaps.

Datasets:
  - cmb: cosmic microwave background temperature
  - complex_arg: phase of complex-valued function on complex plane
  - complex_mag: magnitude of complex-valued function on complex plane
  - kh_bb_ratio: magnetized Kelvin-Helmholtz magnetic field ratio
  - kh_rho: magnetized Kelvin-Helmholtz density
  - ray: synchrotron intensity from ray tracing of GR simulation
  - torus_j: equatorial slice of current density from GR torus
  - torus_rho: poloidal slice of density from GR torus

Colormaps:
  - viridis: built-in, monotonic
  - plasma: built-in, monotonic
  - inferno: built-in, monotonic
  - magma: built-in, monotonic
  - gray_uniform: custom, monotonic
  - cool_uniform: custom, monotonic
  - warm_uniform: custom, monotonic
  - viridis_alt: custom, monotonic
  - plasma_alt: custom, monotonic
  - inferno_alt: custom, monotonic
  - inferno_k: custom, monotonic
  - inferno_w: custom, monotonic
  - inferno_kw: custom, monotonic
  - RdBu: built-in, diverging
  - red_white_blue: custom, diverging
  - red_black_blue: custom, diverging
"""

# Python standard modules
import argparse

# Numerical modules
import numpy as np

# Plotting modules
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Custom color modules
import custom_colormaps

# Main function
def main(**kwargs):

  # Parameters
  data_dir = 'data'
  plot_dir = 'plots'

  # Plotting parameters - layout
  fig_width = 3.35
  aspects = {'cmb': 2.0, 'complex_arg': 2.0, 'complex_mag': 2.0, 'kh_bb_ratio': 2.0, 'kh_rho': 2.0, 'ray': 1.0, 'torus_j': 1.0, 'torus_rho': 1.0}
  lmar_frac = 0.01
  rmar_frac = 0.01
  bmar_frac = 0.01
  tmar_frac = 0.01
  dpi = 300

  # Calculate general layout
  panel_width = fig_width / (lmar_frac + 1.0 + rmar_frac)
  lmar = lmar_frac * panel_width
  rmar = rmar_frac * panel_width
  bmar = bmar_frac * panel_width
  tmar = tmar_frac * panel_width

  # Read data
  data = {}
  if 'cmb' in kwargs['datasets']:
    data_local = np.load('{0}/cmb.npz'.format(data_dir))
    data['cmb'] = dict(data_local)
  if 'complex_arg' in kwargs['datasets'] or 'complex_mag' in kwargs['datasets']:
    data_local = np.load('{0}/complex.npz'.format(data_dir))
    data['complex'] = dict(data_local)
  if 'kh_bb_ratio' in kwargs['datasets'] or 'kh_rho' in kwargs['datasets']:
    data_local = np.load('{0}/kh.npz'.format(data_dir))
    data['kh'] = dict(data_local)
  if 'ray' in kwargs['datasets']:
    data_local = np.load('{0}/ray.npz'.format(data_dir))
    data['ray'] = dict(data_local)
  if 'torus_j' in kwargs['datasets']:
    data_local = np.load('{0}/torus_j.npz'.format(data_dir))
    data['torus_j'] = dict(data_local)
  if 'torus_rho' in kwargs['datasets']:
    data_local = np.load('{0}/torus_rho.npz'.format(data_dir))
    data['torus_rho'] = dict(data_local)

  # Define colormaps
  if 'gray_uniform' in kwargs['colormaps']:
    custom_colormaps.gray_uniform()
  if 'cool_uniform' in kwargs['colormaps']:
    custom_colormaps.cool_uniform()
  if 'warm_uniform' in kwargs['colormaps']:
    custom_colormaps.warm_uniform()
  if 'viridis_alt' in kwargs['colormaps']:
    custom_colormaps.viridis_alt()
  if 'plasma_alt' in kwargs['colormaps']:
    custom_colormaps.plasma_alt()
  if 'inferno_alt' in kwargs['colormaps']:
    custom_colormaps.inferno_alt()
  if 'inferno_k' in kwargs['colormaps']:
    custom_colormaps.inferno_k()
  if 'inferno_w' in kwargs['colormaps']:
    custom_colormaps.inferno_w()
  if 'inferno_kw' in kwargs['colormaps']:
    custom_colormaps.inferno_kw()
  if 'red_white_blue' in kwargs['colormaps']:
    custom_colormaps.red_white_blue()
  if 'red_black_blue' in kwargs['colormaps']:
    custom_colormaps.red_black_blue()

  # Go through datasets and colormaps
  for dataset in kwargs['datasets']:
    for colormap in kwargs['colormaps']:

      # Report status
      print('Plotting dataset {0} with colormap {1}'.format(dataset, colormap))

      # Calculate specific layout
      panel_height = fig_width / aspects[dataset]
      fig_height = bmar + panel_height + tmar

      # Prepare figure
      plt.figure(figsize=(fig_width,fig_height))
      ax = plt.subplot(1, 1, 1)

      # Plot data
      if dataset == 'cmb':
        xf = data['cmb']['xf']
        yf = data['cmb']['yf']
        vals = data['cmb']['temperature']
        ax.pcolormesh(xf, yf, vals, vmin=-6.0e-4, vmax=6.0e-4, cmap=colormap)
      if dataset == 'complex_arg':
        xf = data['complex']['xf']
        yf = data['complex']['yf']
        vals = np.angle(data['complex']['f'])
        ax.pcolormesh(xf, yf, vals, vmin=-np.pi, vmax=np.pi, cmap=colormap)
      if dataset == 'complex_mag':
        xf = data['complex']['xf']
        yf = data['complex']['yf']
        vals = np.abs(data['complex']['f'])
        ax.pcolormesh(xf, yf, vals, vmin=1.0e-4, vmax=1.0e4, norm=LogNorm(), cmap=colormap)
      if dataset == 'kh_bb_ratio':
        xf = data['kh']['xf']
        yf = data['kh']['yf']
        vals = data['kh']['bb_ratio']
        ax.pcolormesh(xf, yf, vals, vmin=1.0e-2, vmax=1.0e0, norm=LogNorm(), cmap=colormap)
      if dataset == 'kh_rho':
        xf = data['kh']['xf']
        yf = data['kh']['yf']
        vals = data['kh']['rho']
        ax.pcolormesh(xf, yf, vals, vmin=0.7, vmax=1.1, cmap=colormap)
      if dataset == 'ray':
        xf = data['ray']['xf']
        yf = data['ray']['yf']
        vals = data['ray']['tt_b']
        ax.pcolormesh(xf, yf, vals, vmin=0.0, vmax=9.0, cmap=colormap)
      if dataset == 'torus_j':
        xf = data['torus_j']['xf']
        yf = data['torus_j']['yf']
        vals = data['torus_j']['j_fluid']
        ax.pcolormesh(xf, yf, vals, vmin=1.0e-3, vmax=1.0e3, norm=LogNorm(), cmap=colormap)
        spin = data['torus_j']['spin']
        r_hor = 1.0 + (1.0**2 - spin**2) ** 0.5
        black_hole = plt.Circle((0.0, 0.0), r_hor, color='k')
        ax.add_artist(black_hole)
      if dataset == 'torus_rho':
        xf = data['torus_rho']['xf']
        yf = data['torus_rho']['yf']
        vals = data['torus_rho']['rho']
        ax.pcolormesh(xf, yf, vals, vmin=1.0e-6, vmax=1.0e0, norm=LogNorm(), cmap=colormap)
        spin = data['torus_rho']['spin']
        r_hor = 1.0 + (1.0**2 - spin**2) ** 0.5
        black_hole = plt.Circle((0.0, 0.0), r_hor, color='k')
        ax.add_artist(black_hole)

      # Adjust axes
      if dataset == 'cmb':
        ax.set_xlim((-2.0**1.5, 2.0**1.5))
        ax.set_ylim((-2.0**0.5, 2.0**0.5))
      if dataset in ('complex_arg', 'complex_mag'):
        ax.set_xlim((-10.0, 10.0))
        ax.set_ylim((-5.0, 5.0))
      if dataset in ('kh_bb_ratio', 'kh_rho'):
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((-0.25, 0.25))
      if dataset == 'ray':
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
      if dataset == 'torus_j':
        ax.set_xlim((-20.0, 20.0))
        ax.set_ylim((-20.0, 20.0))
      if dataset == 'torus_rho':
        ax.set_xlim((-50.0, 50.0))
        ax.set_ylim((-50.0, 50.0))
      ax.axis('off')

      # Adjust layout
      width = panel_width / fig_width
      height = panel_height / fig_height
      x0 = lmar / fig_width
      y0 = bmar / fig_height
      ax.set_position((x0, y0, width, height))

      # Save figure
      plt.savefig('{0}/{1}.{2}.png'.format(plot_dir, dataset, colormap), dpi=dpi)
      plt.close()

# Parser for list of datasets
def dataset_list(string):
  valid_datasets = ['cmb', 'complex_arg', 'complex_mag', 'kh_bb_ratio', 'kh_rho', 'ray', 'torus_j', 'torus_rho']
  if string == 'all':
    return valid_datasets[:]
  selected_datasets = string.split(',')
  for dataset in selected_datasets:
    if dataset not in valid_datasets:
      raise RuntimeError('Invalid dataset: {0}'.format(dataset))
  return selected_datasets

# Parser for list of colormaps
def colormap_list(string):
  valid_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'gray_uniform', 'cool_uniform', 'warm_uniform', 'viridis_alt', 'plasma_alt', 'inferno_alt', 'inferno_k', 'inferno_w', 'inferno_kw', 'RdBu', 'red_white_blue', 'red_black_blue']
  if string == 'all':
    return valid_colormaps[:]
  selected_colormaps = string.split(',')
  for colormap in selected_colormaps:
    if colormap not in valid_colormaps:
      raise RuntimeError('Invalid colormap: {0}'.format(colormap))
  return selected_colormaps

# Execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('datasets', type=dataset_list, help='comma-separated list of datasets to plot')
  parser.add_argument('colormaps', type=colormap_list, help='comma-separated list of colormaps to use')
  args = parser.parse_args()
  main(**vars(args))
