#! /usr/bin/env python

"""
Script for generating and saving complex-valued function data.

f(z) = (z - a)^(1/2) * (z + a)^(-1/2) * (z - z_1) * (z - z_2)^3 / (z - z_3) / (z - z_4)^3
"""

# Modules
import numpy as np

# Main function
def main():

  # Parameters
  output_file = '/Users/cjwhite/projects/colormaps/data/complex.npz'
  extent_x = 10.0
  extent_y = 5.0
  nx = 1024
  ny = 512
  a = 6.0
  z1 = 3.0 + 3.0j
  z2 = -3.0 + 3.0j
  z3 = -3.0 - 3.0j
  z4 = 3.0 - 3.0j

  # Calculate grid
  xf = np.linspace(-extent_x, extent_x, nx + 1)
  yf = np.linspace(-extent_y, extent_y, ny + 1)
  x = 0.5 * (xf[:-1] + xf[1:])
  y = 0.5 * (yf[:-1] + yf[1:])
  z = x[None,:] + y[:,None] * 1.0j

  # Calculate function values
  f1_mag = np.abs(z - a) ** 0.5 * np.abs(z + a) ** -0.5
  f1_arg = 0.5 * (np.angle(z - a) - np.angle(z + a))
  f1 = f1_mag * np.exp(f1_arg * 1.0j)
  f2 = (z - z1) * (z - z2)**3 / (z - z3) / (z - z4)**3
  f = f1 * f2

  # Assemble data
  data_out = {}
  data_out['xf'] = xf
  data_out['yf'] = yf
  data_out['f'] = f

  # Save data
  np.savez(output_file, **data_out)

# Execute main function
if __name__ == '__main__':
  main()
