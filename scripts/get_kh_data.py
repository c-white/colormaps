#! /usr/bin/env python

"""
Script for extracting and saving data from Kelvin-Helmholtz simulation.
"""

# Python standard modules
import sys

# Other modules
import numpy as np

# Main function
def main():

  # Parameters
  read_dir = '/Users/cjwhite/codes/athena/vis/python'
  input_file = '/Users/cjwhite/research/athena_method/data/opt/kh_opt_hlle.prim.00001.athdf'
  output_file = '/Users/cjwhite/projects/colormaps/data/kh_{0}.npz'

  # Load data reader
  sys.path.insert(0, read_dir)
  import athena_read

  # Read data
  data_in = athena_read.athdf(input_file, quantities=('rho','Bcc1','Bcc2','Bcc3'))
  xf = data_in['x1f']
  yf = data_in['x2f']
  rho = data_in['rho'][0,:,:]
  bb1 = data_in['Bcc1'][0,:,:]
  bb2 = data_in['Bcc2'][0,:,:]
  bb3 = data_in['Bcc3'][0,:,:]

  # Calculate derived quantity
  bb_ratio = (bb1 ** 2 + bb2 ** 2) ** 0.5 / bb3

  # Assemble data
  data_out_rho = {}
  data_out_rho['xf'] = xf
  data_out_rho['yf'] = yf
  data_out_rho['rho'] = rho
  data_out_bb_ratio = {}
  data_out_bb_ratio['xf'] = xf
  data_out_bb_ratio['yf'] = yf
  data_out_bb_ratio['bb_ratio'] = bb_ratio

  # Save data
  np.savez(output_file.format('rho'), **data_out_rho)
  np.savez(output_file.format('bb_ratio'), **data_out_bb_ratio)

# Execute main function
if __name__ == '__main__':
  main()
