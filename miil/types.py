#!/usr/bin/env python

import numpy as np

ped_dtype = np.dtype([
    ('name', 'S32'),
    ('events', int),
    ('spatA', float),
    ('spatA_std', float),
    ('spatB', float),
    ('spatB_std', float),
    ('spatC', float),
    ('spatC_std', float),
    ('spatD', float),
    ('spatD_std', float),
    ('comL0', float),
    ('comL0_std', float),
    ('comH0', float),
    ('comH0_std', float),
    ('comL1', float),
    ('comL1_std', float),
    ('comH1', float),
    ('comH1_std', float)])

uv_dtype = np.dtype([
    ('u', float),
    ('v', float)])

loc_dtype = np.dtype([
    ('use', bool),
    ('x', float),
    ('y', float)])

cal_dtype = np.dtype([
    ('use', bool),
    ('x', float),
    ('y', float),
    ('gain_spat', float),
    ('gain_comm', float),
    ('eres_spat', float),
    ('eres_comm', float)])

tcal_dtype = np.dtype([
    ('offset', float),
    ('edep_offset', float)])

eventraw_dtype = np.dtype([
    ('ct', np.int64),
    ('com0', np.int16),
    ('com1', np.int16),
    ('com0h', np.int16),
    ('com1h', np.int16),
    ('u0', np.int16),
    ('v0', np.int16),
    ('u1', np.int16),
    ('v1', np.int16),
    ('u0h', np.int16),
    ('v0h', np.int16),
    ('u1h', np.int16),
    ('v1h', np.int16),
    ('a', np.int16),
    ('b', np.int16),
    ('c', np.int16),
    ('d', np.int16),
    ('panel', np.int8),
    ('cartridge', np.int8),
    ('daq', np.int8),
    ('rena', np.int8),
    ('module', np.int8),
    ('flags', np.int8, (3,))], align=True)

eventcal_dtype = np.dtype([
    ('ct', np.int64),
    ('ft', np.float32),
    ('E', np.float32),
    ('spat_total', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('panel', np.int8),
    ('cartridge', np.int8),
    ('fin', np.int8),
    ('module', np.int8),
    ('apd', np.int8),
    ('crystal', np.int8),
    ('daq', np.int8),
    ('rena', np.int8),
    ('flags', np.int8, (4,))], align=True)

eventcoinc_dtype = np.dtype([
    ('ct0', np.int64),
    ('dtc', np.int64),
    ('ft0', np.float32),
    ('dtf', np.float32),
    ('E0', np.float32),
    ('E1', np.float32),
    ('spat_total0', np.float32),
    ('spat_total1', np.float32),
    ('x0', np.float32),
    ('x1', np.float32),
    ('y0', np.float32),
    ('y1', np.float32),
    ('cartridge0', np.int8),
    ('cartridge1', np.int8),
    ('fin0', np.int8),
    ('fin1', np.int8),
    ('module0', np.int8),
    ('module1', np.int8),
    ('apd0', np.int8),
    ('apd1', np.int8),
    ('crystal0', np.int8),
    ('crystal1', np.int8),
    ('daq0', np.int8),
    ('daq1', np.int8),
    ('rena0', np.int8),
    ('rena1', np.int8),
    ('flags', np.int8, (2,))], align=True)

cudarecon_type0_vec_dtype = np.dtype([
    ('pos0', np.float32, (3,)),
    ('dt', np.float32),
    ('randoms_est', np.float32),
    ('pos1', np.float32, (3,)),
    ('tof_scatter_est', np.float32),
    ('scatter_est', np.float32)], align=True)

cudarecon_type1_vec_dtype = np.dtype([
    ('pos0', np.float32, (3,)),
    ('weight', np.float32),
    ('e0', np.float32),  # Appears to be unused
    ('pos1', np.float32, (3,)),
    ('weight1', np.float32),  # Appears to be unused
    ('e1', np.float32)], align=True) # Appears to be unused

cuda_vox_file_header_dtype = np.dtype([
    ('magic_number', np.int32),
    ('version_number', np.int32),
    ('size', np.int32, (3,))], align=True)
