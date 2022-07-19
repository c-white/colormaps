#! /usr/bin/env python

"""
Script for exporting all custom colormaps into Python modules.

Colormaps:
  - gray_uniform: custom, monotonic
  - cool_uniform: custom, monotonic
  - warm_uniform: custom, monotonic
  - viridis_alt: custom, monotonic
  - plasma_alt: custom, monotonic
  - inferno_alt: custom, monotonic
  - inferno_k: custom, monotonic
  - inferno_w: custom, monotonic
  - inferno_kw: custom, monotonic
  - red_white_blue: custom, diverging
  - red_black_blue: custom, diverging
"""

# Python standard modules
import argparse

# Color modules
import matplotlib.cm as cm

# Custom color modules
import custom_colormaps

# Main function
def main(**kwargs):

  # Parameters
  export_file = 'scripts/custom_colormap_data.py'

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

  # Open file for writing
  with open(export_file, 'w') as f:

    # Write header
    f.write('import matplotlib.cm as cm\n')
    f.write('import matplotlib.colors as colors\n')

    # Go through colormaps
    for name in kwargs['colormaps']:

      # Extract data
      cmap = cm.get_cmap(name)
      red = cmap._segmentdata['red']
      green = cmap._segmentdata['green']
      blue = cmap._segmentdata['blue']

      # Write RGB data
      f.write('\n')
      f.write('{0}_red = [\n'.format(name))
      for vals in red[:-1]:
        f.write('  {0},\n'.format(list(vals)))
      f.write('  {0}]\n'.format(list(red[-1])))
      f.write('{0}_green = [\n'.format(name))
      for vals in green[:-1]:
        f.write('  {0},\n'.format(list(vals)))
      f.write('  {0}]\n'.format(list(green[-1])))
      f.write('{0}_blue = [\n'.format(name))
      for vals in blue[:-1]:
        f.write('  {0},\n'.format(list(vals)))
      f.write('  {0}]\n'.format(list(blue[-1])))

      # Assemble and register colormap
      f.write('{0}_data = '.format(name) + '{' + '\'red\': {0}_red, \'green\': {0}_green, \'blue\': {0}_blue'.format(name) + '}\n')
      f.write('{0}_cmap = colors.LinearSegmentedColormap(\'{0}\', {0}_data, N={1})\n'.format(name, cmap.N))
      f.write('cm.register_cmap(name=\'{0}\', cmap={0}_cmap)\n'.format(name))

# Parser for list of colormaps
def colormap_list(string):
  valid_colormaps = ['gray_uniform', 'cool_uniform', 'warm_uniform', 'viridis_alt', 'plasma_alt', 'inferno_alt', 'inferno_k', 'inferno_w', 'inferno_kw', 'red_white_blue', 'red_black_blue']
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
  parser.add_argument('colormaps', type=colormap_list, help='comma-separated list of colormaps to use')
  args = parser.parse_args()
  main(**vars(args))
