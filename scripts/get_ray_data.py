#! /usr/bin/env python

"""
Script for extracting and saving ray tracing data.
"""

# Python standard modules
import sys
import warnings

# Other modules
import healpy
import numpy as np

# Main function
def main():

  # Parameters
  input_file = '/Users/cjwhite/research/notes/art/data/rays/rays_{0}{1}_40.dat'
  output_file = 'data/ray.npz'
  input_format = np.float64
  output_format = np.float32
  n_small = 512
  n_large = 1024
  offsets = (0, 1)
  letter_vals = (('b', 'c'), ('f', 'g'))

  # Read image data
  tt_b = np.zeros((n_large, n_large), dtype=input_format)
  for x_offset in offsets:
    for y_offset in offsets:
      for letter_1 in letter_vals[x_offset]:
        for letter_2 in letter_vals[y_offset]:
          tt_b[x_offset::2,y_offset::2] += np.reshape(np.fromfile(input_file.format(letter_1, letter_2), dtype=input_format), (n_small, n_small))

  # Process image data
  tt_b = np.where(np.isnan(tt_b), 0.0, tt_b)
  tt_b /= 4.0e10
  tt_b = tt_b.astype(output_format)

  # Assemble data
  data = {}
  data['xf'] = np.linspace(0.0, 1.0, n_large + 1)
  data['yf'] = np.linspace(0.0, 1.0, n_large + 1)
  data['tt_b'] = tt_b

  # Save data
  np.savez(output_file, **data)

# Execute main function
if __name__ == '__main__':
  main()
