#!/usr/bin/env python

import numpy as np
from defaults import *

# For Calibrated Events
def get_global_cartridge_number(events, system_shape=default_system_shape):
    '''
    Takes eventcal_dtype events and returns the global catridge number for each
    event given system_shape.
    '''
    global_cartridge = events['cartridge'].astype(int) + \
                       system_shape[1] * events['panel'].astype(int)
    return global_cartridge

def get_global_fin_number(events, system_shape=default_system_shape):
    '''
    Takes eventcal_dtype events and returns the global fin number for each
    event given system_shape.  Uses get_global_cartridge_number as a base.
    '''
    global_cartridge = get_global_cartridge_number(events, system_shape)
    global_fin = events['fin'] + system_shape[2] * global_cartridge
    return global_fin

def get_global_module_number(events, system_shape=default_system_shape):
    '''
    Takes eventcal_dtype events and returns the global module number for each
    event given system_shape.  Uses get_global_fin_number as a base.
    '''
    global_fin = get_global_fin_number(events, system_shape)
    global_module = events['module'] + system_shape[3] * global_fin
    return global_module

def get_global_apd_number(events, system_shape=default_system_shape):
    '''
    Takes eventcal_dtype events and returns the global apd number for each
    event given system_shape.  Uses get_global_module_number as a base.
    '''
    global_module = get_global_module_number(events, system_shape)
    global_apd = events['apd'] + system_shape[4] * global_module
    return global_apd

def get_global_crystal_number(events, system_shape=default_system_shape):
    '''
    Takes eventcal_dtype events and returns the global crystal number for each
    event given system_shape.  Uses get_global_apd_number as a base.
    '''
    global_apd = get_global_apd_number(events, system_shape)
    global_crystal = events['crystal'] + system_shape[5] * global_apd
    return global_crystal

# For Coincidence Events
def get_global_cartridge_numbers(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    cartridge number for each event given system_shape.
    '''
    global_cartridge0 = events['cartridge0'].astype(int)
    global_cartridge1 = events['cartridge1'].astype(int) + system_shape[1]
    return global_cartridge0, global_cartridge1

def get_global_fin_numbers(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    fin number for each event given system_shape.  Uses
    get_global_cartridge_numbers as a base.
    '''
    global_cartridge0, global_cartridge1 = \
            get_global_cartridge_numbers(events, system_shape)
    global_fin0 = events['fin0'] + system_shape[2] * global_cartridge0
    global_fin1 = events['fin1'] + system_shape[2] * global_cartridge1
    return global_fin0, global_fin1

def get_global_module_numbers(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    module number for each event given system_shape.  Uses
    get_global_fin_numbers as a base.
    '''
    global_fin0, global_fin1 = get_global_fin_numbers(events, system_shape)
    global_module0 = events['module0'] + system_shape[3] * global_fin0
    global_module1 = events['module1'] + system_shape[3] * global_fin1
    return global_module0, global_module1

def get_global_apd_numbers(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    apd number for each event given system_shape.  Uses
    get_global_module_numbers as a base.
    '''
    global_module0, global_module1 = \
            get_global_module_numbers(events, system_shape)
    global_apd0 = events['apd0'] + system_shape[4] * global_module0
    global_apd1 = events['apd1'] + system_shape[4] * global_module1
    return global_apd0, global_apd1

def get_global_crystal_numbers(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    crystal number for each event given system_shape.  Uses
    get_global_apd_numbers as a base.
    '''
    global_apd0, global_apd1 = get_global_apd_numbers(events, system_shape)
    global_crystal0 = events['crystal0'] + system_shape[5] * global_apd0
    global_crystal1 = events['crystal1'] + system_shape[5] * global_apd1
    return global_crystal0, global_crystal1

def get_global_lor_number(events, system_shape=default_system_shape):
    '''
    Takes eventcoinc_dtype events and returns the global lor number for each
    event given system_shape.  Uses get_global_crystal_numbers as a base.
    Global lor calculated as:
        (global_crystal0 * no_crystals_per_panel) +
        (global_crystal1 - no_crystals_per_panel)
    '''
    global_crystal0, global_crystal1 = \
            get_global_crystal_numbers(events, system_shape)
    no_crystals_per_panel = np.prod(system_shape[1:])

    return (global_crystal0 * no_crystals_per_panel) + \
           (global_crystal1 - no_crystals_per_panel)

def get_crystals_from_lor(lors, system_shape=default_system_shape):
    '''
    Takes an array of lor indices and returns the left and right global crystal
    number based on the given system shape.
    '''
    crystal0 = lors // np.prod(system_shape[1:])
    crystal1 = lors % np.prod(system_shape[1:]) + np.prod(system_shape[1:])
    return crystal0, crystal1

def get_apds_from_lor(lors, system_shape=default_system_shape):
    '''
    Takes an array of lor indices and returns the left and right global apd
    number based on the given system shape.  Uses get_crystals_from_lor as a
    base for calculation.
    '''
    crystal0, crystal1 = get_crystals_from_lor(lors, system_shape)
    apd0 = crystal0 // system_shape[5]
    apd1 = crystal1 // system_shape[5]
    return apd0, apd1

def get_modules_from_lor(lors, system_shape=default_system_shape):
    '''
    Takes an array of lor indices and returns the left and right global module
    number based on the given system shape.  Uses get_apds_from_lor as a base
    for calculation.
    '''
    apd0, apd1 = get_apds_from_lor(lors, system_shape)
    module0 = apd0 // system_shape[4]
    module1 = apd1 // system_shape[4]
    return module0, module1

def lor_to_slor(
        lors,
        slor_shape = default_slor_shape,
        system_shape = default_system_shape):
    '''
    Breaks down an array of LOR indices and transforms them into SLOR indices.
    SLORs, or symmetric LORs, is a way of indicating LORs that see the same
    attenuation and are rotationaly and/or translationally symmetric, assuming
    a symmetric source.

    Slors are effecitvely addressed by their place in a five dimensional array.
    [apd_sum][module_diff][fin_diff][crystal0][crystal1].

    This does not fully exploit the symmetry for a module difference of 0, but
    it is much easier to ignore this, rather than have SLOR indices that cannot
    be generated, nor have a special case.
    '''
    no_crystal_per_panel = np.prod(miil.default_system_shape[1:])
    crystals_per_fin = np.prod(miil.default_system_shape[3:])
    crystals_per_module = np.prod(miil.default_system_shape[4:])
    crystals_per_apd = miil.default_system_shape[5]
    apds_per_module = miil.default_system_shape[5]

    c0 = lors // no_crystal_per_panel
    c1 = lors % no_crystal_per_panel

    apd0 = c0 // crystals_per_apd
    apd1 = c1 // crystals_per_apd
    apd_sum = (apd0 % miil.default_system_shape[4]) + \
              (apd1 % miil.default_system_shape[4])
    del apd0, apd1
    slors = apd_sum * slor_shape[1]
    del apd_sum

    m0 = c0 // crystals_per_module
    m1 = c1 // crystals_per_module
    mod_diff = np.abs((m0 % miil.default_system_shape[3]) -
                      (m1 % miil.default_system_shape[3]))
    del m0, m1
    slors = (slors + mod_diff) * slor_shape[2]
    del mod_diff

    f0 = c0 // crystals_per_fin
    f1 = c1 // crystals_per_fin
    fin_diff = np.abs(f0 - f1)
    del f0, f1
    slors = (slors + fin_diff) * slor_shape[3]
    del fin_diff

    slors = (slors + (c0 % crystals_per_apd)) * slor_shape[4]
    del c0
    slors += (c1 % crystals_per_apd)
    del c1
    return slors

def lor_to_slor_bins(
        lors,
        slor_shape = default_slor_shape,
        system_shape = default_system_shape):
    '''
    Converts LORs to SLORs using lor_to_slor and then bins them using
    numpy.bincount.

    Returns an array, shape = (np.prod(slor_shape),), representing the number of
    LORs with that SLOR index.
    '''
    slors = lor_to_slor(lors, slor_shape)
    bins = np.bincount(slors, minlength=np.prod(slor_shape))
    return bins
