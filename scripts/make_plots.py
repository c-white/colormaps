#! /usr/bin/env python

"""
Script for making all plots.

Datasets:
  - complex: complex-valued function on complex plane
  - torus_rho: poloidal slice of density from GR torus
  - magnetized Kelvin-Helmholtz density
  - magnetized Kelvin-Helmholtz magnetization
  - equatorial slice of current density from GR torus
  - cosmic microwave background

Colormaps:
  - viridis
"""

# Python standard modules
import argparse

# Python plotting modules
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters
  data_dir = '/Users/cjwhite/projects/colormaps/data'
  plot_dir = '/Users/cjwhite/projects/colormaps/plots'

  # Plotting parameters - layout
  fig_width = 3.35
  aspects = {'complex': 2.0, 'torus_rho': 1.0}
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
  if 'complex' in kwargs['datasets']:
    data_local = np.load('{0}/complex.npz'.format(data_dir))
    data['complex'] = dict(data_local)
  if 'torus_rho' in kwargs['datasets']:
    data_local = np.load('{0}/torus.npz'.format(data_dir))
    data['torus'] = dict(data_local)

  # Go through datasets
  for dataset in kwargs['datasets']:

    # Calculate specific layout
    panel_height = fig_width / aspects[dataset]
    fig_height = bmar + panel_height + tmar

    # Prepare figure
    plt.figure(figsize=(fig_width,fig_height))
    ax = plt.subplot(1, 1, 1)

    # Plot data
    if dataset == 'complex':
      xf = data['complex']['xf']
      yf = data['complex']['yf']
      vals = np.angle(data['complex']['f'])
      ax.pcolormesh(xf, yf, vals, vmin=-np.pi, vmax=np.pi)
    if dataset == 'torus_rho':
      xf = data['torus']['xf']
      yf = data['torus']['yf']
      vals = data['torus']['rho']
      ax.pcolormesh(xf, yf, vals, vmin=1.0e-5, vmax=1.0e0, norm=LogNorm())
      spin = data['torus']['spin']
      r_hor = 1.0 + (1.0**2 - spin**2) ** 0.5
      black_hole = plt.Circle((0.0, 0.0), r_hor, color='k')
      ax.add_artist(black_hole)

    # Adjust axes
    if dataset == 'complex':
      ax.set_xlim((-10.0, 10.0))
      ax.set_ylim((-5.0, 5.0))
    if dataset == 'torus_rho':
      ax.set_xlim((-50.0, 50.0))
      ax.set_ylim((-50.0, 50.0))
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Adjust layout
    width = panel_width / fig_width
    height = panel_height / fig_height
    x0 = lmar / fig_width
    y0 = bmar / fig_height
    ax.set_position((x0, y0, width, height))

    # Go through colormaps
    for colormap in kwargs['colormaps']:

      # Set colormap
      plt.set_cmap(colormap)

      # Save figure
      plt.savefig('{0}/{1}.{2}.png'.format(plot_dir, dataset, colormap), dpi=dpi)

    # Prepare for next dataset
    plt.close()

# Parser for list of datasets
def dataset_list(string):
  valid_datasets = ['complex', 'torus_rho']
  if string == 'all':
    return valid_datasets[:]
  selected_datasets = string.split(',')
  for dataset in selected_datasets:
    if dataset not in valid_datasets:
      raise RuntimeError('Invalid dataset: {0}'.format(dataset))
  return selected_datasets

# Parser for list of colormaps
def colormap_list(string):
  valid_colormaps = ['viridis']
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
