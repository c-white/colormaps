#! /usr/bin/env python

"""
Script for plotting all colorbars.

Colormaps:
  - viridis
  - gray_uniform
  - cool_uniform
"""

# Python plotting modules
import matplotlib
matplotlib.use('agg')
matplotlib.rc('font', size=8, family='serif')
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# Modules
import numpy as np
import custom_colormaps

# Main function
def main():

  # Parameters
  data_dir = '/Users/cjwhite/projects/colormaps/data'
  plot_dir = '/Users/cjwhite/projects/colormaps/plots'
  cmaps = ['viridis', 'gray_uniform', 'cool_uniform']

  # Plotting parameters - layout
  fig_width = 3.35
  lmar_frac = 0.05
  rmar_frac = 0.05
  bmar_frac = 0.05
  tmar_frac = 0.15
  vmar_frac = 0.1
  hmar_frac = 0.15
  cbar_height_frac = 0.05
  num_cols = 2
  dpi = 300

  # Calculate general layout
  cbar_width = fig_width / (lmar_frac + num_cols + (num_cols - 1) * vmar_frac + rmar_frac)
  lmar = lmar_frac * cbar_width
  rmar = rmar_frac * cbar_width
  bmar = bmar_frac * cbar_width
  tmar = tmar_frac * cbar_width
  vmar = vmar_frac * cbar_width
  hmar = hmar_frac * cbar_width
  cbar_height = cbar_height_frac * cbar_width
  num_rows = (len(cmaps) + num_cols - 1) / num_cols
  fig_height = bmar + num_rows * cbar_height + (num_rows - 1) * hmar + tmar

  # Define colormaps
  custom_colormaps.gray_uniform()
  custom_colormaps.cool_uniform()

  # Prepare figure
  plt.figure(figsize=(fig_width,fig_height))
  axes = []
  for n in range(1, num_rows * num_cols + 1):
      axes.append(plt.subplot(num_rows, num_cols, n))

  # Make colorbars
  norm = Normalize(vmin=0.0, vmax=1.0)
  for ax, cmap in zip(axes, cmaps):
    ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_xlabel(cmap)
    ax.xaxis.set_label_position('top')
  for ax in axes[len(cmaps):]:
    ax.set_visible(False)

  # Adjust layout
  width = cbar_width / fig_width
  height = cbar_height / fig_height
  n = 0
  for row in range(num_rows):
    for col in range(num_cols):
      x0 = (lmar + col * (cbar_width + vmar)) / fig_width
      y0 = (bmar + (num_rows - row - 1) * (cbar_height + hmar)) / fig_height
      axes[n].set_position((x0, y0, width, height))
      n += 1

  # Save figure
  plt.savefig('{0}/colorbars.png'.format(plot_dir), dpi=dpi)
  plt.close()

# Execute main function
if __name__ == '__main__':
  main()
