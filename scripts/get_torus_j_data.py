#! /usr/bin/env python

"""
Script for extracting and saving current data from GR torus simulation.
"""

# Python standard modules
import sys

# Other modules
import numpy as np

# Main function
def main():

  # Parameters
  read_dir = '/Users/cjwhite/codes/athena/vis/python'
  input_file = '/Users/cjwhite/research/ir_centroid/data/raw/mad_ppm_3.{0}.02001.athdf'
  output_file = '/Users/cjwhite/projects/colormaps/data/torus_j.npz'
  spin = 0.98

  # Load data reader
  sys.path.insert(0, read_dir)
  import athena_read

  # Read data
  quantities_prim = ('vel1', 'vel2', 'vel3')
  quantities_user = ('F01', 'F02', 'F03', 'F12', 'F13', 'F23', 'd_0_F01', 'd_0_F02', 'd_0_F03')
  data_coord = athena_read.athdf(input_file.format('prim'), quantities=[])
  r = data_coord['x1v']
  rf = data_coord['x1f']
  th = data_coord['x2v']
  ph = data_coord['x3v']
  phf = data_coord['x3f']
  nth = len(th)
  if nth % 2 != 0:
    raise RuntimeError('must have even number of cells in theta-direction')
  dth = np.pi / nth
  th_min = np.pi / 2.0 - dth / 2.0
  th_max = np.pi / 2.0 + dth / 2.0
  data_prim = athena_read.athdf(input_file.format('prim'), quantities=quantities_prim, x2_min=th_min, x2_max=th_max)
  data_user = athena_read.athdf(input_file.format('user'), quantities=quantities_user, x2_min=th_min, x2_max=th_max)
  th = th[nth/2-1:nth/2+1]

  # Calculate grids
  xf = rf[None,:] * np.cos(phf[:,None])
  yf = rf[None,:] * np.sin(phf[:,None])

  # Calculate geometry
  sigma = r**2
  alpha = (1.0 + 2.0 * r / sigma) ** -0.5
  g_00 = -(1.0 - 2.0 * r / sigma)
  g_01 = 2.0 * r / sigma
  g_03 = -2.0 * spin * r / sigma
  g_11 = 1.0 + 2.0 * r / sigma
  g_13 = -(1.0 + 2.0 * r / sigma) * spin
  g_22 = sigma
  g_33 = r**2 + spin**2 + 2.0 * spin**2 * r / sigma
  g01 = 2.0 * r / sigma
  det = sigma

  # Extract quantities
  uu1 = data_prim['vel1']
  uu2 = data_prim['vel2']
  uu3 = data_prim['vel3']
  ff01 = data_user['F01']
  ff02 = data_user['F02']
  ff03 = data_user['F03']
  ff12 = data_user['F12']
  ff13 = data_user['F13']
  ff23 = data_user['F23']
  d_0_ff01 = data_user['d_0_F01']
  d_0_ff02 = data_user['d_0_F02']
  d_0_ff03 = data_user['d_0_F03']

  # Calculate velocity
  uu1 = np.mean(uu1, axis=1)
  uu2 = np.mean(uu2, axis=1)
  uu3 = np.mean(uu3, axis=1)
  gamma = (1.0 + g_11[None,:] * uu1**2 + 2.0 * g_13[None,:] * uu1 * uu3 + g_22[None,:] * uu2**2 + g_33[None,:] * uu3**2) ** 0.5
  u0 = gamma / alpha[None,:]
  u1 = uu1 - alpha * g01[None,:] * gamma
  u2 = uu2
  u3 = uu3

  # Calculate field time derivatives
  d_0_ff01 = np.mean(d_0_ff01, axis=1)
  d_0_ff02 = np.mean(d_0_ff02, axis=1)
  d_0_ff03 = np.mean(d_0_ff03, axis=1)

  # Calculate field radial derivatives
  d_1_ff10 = np.diff(-ff01, axis=2) / np.diff(r)[None,None,:]
  d_1_ff12 = np.diff(ff12, axis=2) / np.diff(r)[None,None,:]
  d_1_ff13 = np.diff(ff13, axis=2) / np.diff(r)[None,None,:]
  d_1_ff10 = np.concatenate((d_1_ff10[:,:,:1], d_1_ff10, d_1_ff10[:,:,-1:]), axis=2)
  d_1_ff12 = np.concatenate((d_1_ff12[:,:,:1], d_1_ff12, d_1_ff12[:,:,-1:]), axis=2)
  d_1_ff13 = np.concatenate((d_1_ff13[:,:,:1], d_1_ff13, d_1_ff13[:,:,-1:]), axis=2)
  d_1_ff10 = np.mean(0.5 * (d_1_ff10[:,:,:-1] + d_1_ff10[:,:,1:]), axis=1)
  d_1_ff12 = np.mean(0.5 * (d_1_ff12[:,:,:-1] + d_1_ff12[:,:,1:]), axis=1)
  d_1_ff13 = np.mean(0.5 * (d_1_ff13[:,:,:-1] + d_1_ff13[:,:,1:]), axis=1)

  # Calculate field polar derivatives
  d_2_ff20 = (np.diff(-ff02, axis=1) / np.diff(th)[None,:,None])[:,0,:]
  d_2_ff21 = (np.diff(-ff12, axis=1) / np.diff(th)[None,:,None])[:,0,:]
  d_2_ff23 = (np.diff(ff23, axis=1) / np.diff(th)[None,:,None])[:,0,:]

  # Calculate field azimuthal derivatives
  ph_ext = np.concatenate((ph[-1:] - 2.0*np.pi, ph, ph[:1] + 2.0*np.pi))
  ff30 = np.concatenate((-ff03[-1:,:,:], -ff03, -ff03[:1,:,:]), axis=0)
  ff31 = np.concatenate((-ff13[-1:,:,:], -ff13, -ff13[:1,:,:]), axis=0)
  ff32 = np.concatenate((-ff23[-1:,:,:], -ff23, -ff23[:1,:,:]), axis=0)
  d_3_ff30 = np.diff(ff30, axis=0) / np.diff(ph_ext)[:,None,None]
  d_3_ff31 = np.diff(ff31, axis=0) / np.diff(ph_ext)[:,None,None]
  d_3_ff32 = np.diff(ff32, axis=0) / np.diff(ph_ext)[:,None,None]
  d_3_ff30 = np.mean(0.5 * (d_3_ff30[:-1,:,:] + d_3_ff30[1:,:,:]), axis=1)
  d_3_ff31 = np.mean(0.5 * (d_3_ff31[:-1,:,:] + d_3_ff31[1:,:,:]), axis=1)
  d_3_ff32 = np.mean(0.5 * (d_3_ff32[:-1,:,:] + d_3_ff32[1:,:,:]), axis=1)

  # Calculate current
  j0 = (d_1_ff10 + d_2_ff20 + d_3_ff30) / det[None,:]
  j1 = (d_0_ff01 + d_2_ff21 + d_3_ff31) / det[None,:]
  j2 = (d_0_ff02 + d_1_ff12 + d_3_ff32) / det[None,:]
  j3 = (d_0_ff03 + d_1_ff13 + d_2_ff23) / det[None,:]
  j_0 = g_00[None,:] * j0 + g_01[None,:] * j1 + g_03[None,:] * j3
  j_1 = g_01[None,:] * j0 + g_11[None,:] * j1 + g_13[None,:] * j3
  j_2 = g_22[None,:] * j2
  j_3 = g_03[None,:] * j0 + g_13[None,:] * j1 + g_33[None,:] * j3
  j_sq = j_0 * j0 + j_1 * j1 + j_2 * j2 + j_3 * j3
  j_u = j_0 * u0 + j_1 * u1 + j_2 * u2 + j_3 * u3
  j_fluid = j_sq + (j_u)**2

  # Assemble data
  data_out = {}
  data_out['xf'] = xf
  data_out['yf'] = yf
  data_out['j_fluid'] = j_fluid
  data_out['spin'] = spin

  # Save data
  np.savez(output_file, **data_out)

# Execute main function
if __name__ == '__main__':
  main()
