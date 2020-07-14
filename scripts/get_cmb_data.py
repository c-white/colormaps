#! /usr/bin/env python

"""
Script for extracting and saving Planck CMB data.
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
  input_file = '/Users/cjwhite/research/presentations/2017_03_berkeley/data/cmb.fits'
  output_file = '/Users/cjwhite/projects/colormaps/data/cmb.npz'
  num_x = 1024
  num_y = 512

  # Load data
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'No INDXSCHM keyword in header file', UserWarning, 'healpy')
    with open('/dev/null', 'w') as sys.stdout:
      cmb_map = healpy.read_map(input_file)
    sys.stdout = sys.__stdout__

  # Remap data
  xf = np.linspace(-2.0**1.5, 2.0**1.5, num_x + 1)[None,:]
  yf = np.linspace(-2.0**0.5, 2.0**0.5, num_y + 1)[:,None]
  x = 0.5 * (xf[:,:-1] + xf[:,1:])
  y = 0.5 * (yf[:-1,:] + yf[1:,:])
  psi = np.arcsin(y / 2.0**0.5)
  lat = np.arcsin((2.0 * psi + np.sin(2.0 * psi)) / np.pi)
  lon = np.pi + np.pi * x / (2.0**1.5 * np.cos(psi))
  lon = np.where(abs(lon - np.pi) > np.pi, np.nan, lon)
  th = np.pi / 2.0 - lat
  ph = np.pi - lon
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'invalid value encountered in less', RuntimeWarning)
    ph = np.where(ph < 0.0, ph + 2.0 * np.pi, ph)
  n_pix = cmb_map.shape[0]
  n_side = healpy.npix2nside(n_pix)
  temperature = np.ones((num_y, num_x), dtype=np.float32) * np.nan
  for j in range(num_y):
    for i in range(num_x):
      th_val = th[j]
      ph_val = ph[j,i]
      if np.isfinite(ph_val):
        temperature[j,i] = healpy.get_interp_val(cmb_map, th_val, ph_val)

  # Assemble data
  data = {}
  data['xf'] = xf
  data['yf'] = yf
  data['temperature'] = temperature

  # Save data
  np.savez(output_file, **data)

# Execute main function
if __name__ == '__main__':
  main()
