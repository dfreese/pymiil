#!/usr/bin/env python

import numpy as np
from miil.defaults import default_system_shape, default_slor_shape

# For Calibrated Events


def get_global_cartridge_number(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global catridge number for each
    event given system_shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge = events['cartridge'].astype(int) + \
        system_shape[1] * events['panel'].astype(int)
    return global_cartridge


def get_global_fin_number(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global fin number for each
    event given system_shape.  Uses get_global_cartridge_number as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge = get_global_cartridge_number(events, system_shape)
    global_fin = events['fin'] + system_shape[2] * global_cartridge
    return global_fin


def get_global_module_number(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global module number for each
    event given system_shape.  Uses get_global_fin_number as a base.

    default_system_shape is used if system_shape is None.
    '''
    global_fin = get_global_fin_number(events, system_shape)
    global_module = events['module'] + system_shape[3] * global_fin
    return global_module


def get_global_apd_number(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global apd number for each
    event given system_shape.  Uses get_global_module_number as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_module = get_global_module_number(events, system_shape)
    global_apd = events['apd'] + system_shape[4] * global_module
    return global_apd


def get_global_crystal_number(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global crystal number for each
    event given system_shape.  Uses get_global_apd_number as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_apd = get_global_apd_number(events, system_shape)
    global_crystal = events['crystal'] + system_shape[5] * global_apd
    return global_crystal

# For Coincidence Events


def get_global_cartridge_numbers(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    cartridge number for each event given system_shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge0 = events['cartridge0'].astype(int)
    global_cartridge1 = events['cartridge1'].astype(int) + system_shape[1]
    return global_cartridge0, global_cartridge1


def get_global_fin_numbers(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    fin number for each event given system_shape.  Uses
    get_global_cartridge_numbers as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge0, global_cartridge1 = \
        get_global_cartridge_numbers(events, system_shape)
    global_fin0 = events['fin0'] + system_shape[2] * global_cartridge0
    global_fin1 = events['fin1'] + system_shape[2] * global_cartridge1
    return global_fin0, global_fin1


def get_global_module_numbers(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    module number for each event given system_shape.  Uses
    get_global_fin_numbers as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_fin0, global_fin1 = get_global_fin_numbers(events, system_shape)
    global_module0 = events['module0'] + system_shape[3] * global_fin0
    global_module1 = events['module1'] + system_shape[3] * global_fin1
    return global_module0, global_module1


def get_global_apd_numbers(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    apd number for each event given system_shape.  Uses
    get_global_module_numbers as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_module0, global_module1 = \
        get_global_module_numbers(events, system_shape)
    global_apd0 = events['apd0'] + system_shape[4] * global_module0
    global_apd1 = events['apd1'] + system_shape[4] * global_module1
    return global_apd0, global_apd1


def get_global_crystal_numbers(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    crystal number for each event given system_shape.  Uses
    get_global_apd_numbers as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_apd0, global_apd1 = get_global_apd_numbers(events, system_shape)
    global_crystal0 = events['crystal0'] + system_shape[5] * global_apd0
    global_crystal1 = events['crystal1'] + system_shape[5] * global_apd1
    return global_crystal0, global_crystal1


def get_global_lor_number(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the global lor number for each
    event given system_shape.  Uses get_global_crystal_numbers as a base.
    Global lor calculated as:
        (global_crystal0 * no_crystals_per_panel) +
        (global_crystal1 - no_crystals_per_panel)

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_crystal0, global_crystal1 = \
        get_global_crystal_numbers(events, system_shape)
    no_crystals_per_panel = np.prod(system_shape[1:])

    return (global_crystal0 * no_crystals_per_panel) + \
           (global_crystal1 - no_crystals_per_panel)


def get_crystals_from_lor(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global crystal
    number based on the given system shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0 = lors // np.prod(system_shape[1:])
    crystal1 = lors % np.prod(system_shape[1:]) + np.prod(system_shape[1:])
    return crystal0, crystal1


def get_apds_from_lor(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global apd
    number based on the given system shape.  Uses get_crystals_from_lor as a
    base for calculation.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0, crystal1 = get_crystals_from_lor(lors, system_shape)
    apd0 = crystal0 // system_shape[5]
    apd1 = crystal1 // system_shape[5]
    return apd0, apd1


def get_modules_from_lor(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global module
    number based on the given system shape.  Uses get_apds_from_lor as a base
    for calculation.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    apd0, apd1 = get_apds_from_lor(lors, system_shape)
    module0 = apd0 // system_shape[4]
    module1 = apd1 // system_shape[4]
    return module0, module1


def crystals_to_lor(crystal0, crystal1, system_shape=None, local_id=False):
    '''
    Converts two crystal indices to an lor number.  If local_id is False
    (default), it assumes the crystals are globablly indexed.  If True, it
    assumes they're indexed local to the panel.

    Lors are calculated as follows:
        lor = panel_crystal0 * no_crystals_per_panel + panel_crystal1

    default_slor_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    no_crystals_per_panel = np.prod(system_shape[1:])
    crystal0 = np.asarray(crystal0)
    crystal1 = np.asarray(crystal1)
    if not local_id:
        crystal1 -= no_crystals_per_panel
    lor = crystal0 * no_crystals_per_panel + crystal1
    return lor


def lor_to_slor(lors, slor_shape=None, system_shape=None):
    '''
    Breaks down an array of LOR indices and transforms them into SLOR indices.
    SLORs, or symmetric LORs, is a way of indicating LORs that see the same
    attenuation and are reflectively, rotationally, translationally symmetric,
    assuming a symmetric source.

    Slors are effecitvely addressed by their place in a five dimensional array.
    [fin_diff][near_x][x_diff][near_y][far_y].

    default_slor_shape is used if system_shape is None.

    default_system_shape is used if system_shape is None.
    '''
    if slor_shape is None:
        slor_shape = default_slor_shape
    if system_shape is None:
        system_shape = default_system_shape

    no_crystal_per_panel = np.prod(system_shape[1:])
    crystals_per_fin = np.prod(system_shape[3:])
    crystals_per_module = np.prod(system_shape[4:])
    crystals_per_apd = system_shape[5]

    crystal0 = lors // no_crystal_per_panel
    crystal1 = lors % no_crystal_per_panel

    fin0 = crystal0 // crystals_per_fin
    fin1 = crystal1 // crystals_per_fin
    fin_diff = np.abs(fin0 - fin1)
    del fin0, fin1
    if (fin_diff >= slor_shape[0]).any():
        raise ValueError("fin_diff out of range")
    slors = fin_diff * slor_shape[1]
    del fin_diff

    apd0 = (crystal0 // crystals_per_apd) % system_shape[4]
    apd1 = (crystal1 // crystals_per_apd) % system_shape[4]

    y_crystal_near = 7 - (crystal0 % 8) + 8 * apd0;
    y_crystal_far = 7 - (crystal1 % 8) + 8 * apd1;
    del apd0, apd1

    x_local_crystal_near = (crystal0 % crystals_per_apd) // 8
    x_local_crystal_far = 7 - (crystal1 % crystals_per_apd) // 8

    module0 = (crystal0 // crystals_per_module) % system_shape[3]
    module1 = (crystal1 // crystals_per_module) % system_shape[3]
    x_crystal_near = 8 * module0 + x_local_crystal_near
    x_crystal_far = 8 * module1 + x_local_crystal_far
    del module0, module1

    mirror_y = y_crystal_near > y_crystal_far
    y_crystal_near[mirror_y], y_crystal_far[mirror_y] = \
            y_crystal_far[mirror_y], y_crystal_near[mirror_y].copy()
    x_crystal_near[mirror_y], x_crystal_far[mirror_y] = \
            x_crystal_far[mirror_y], x_crystal_near[mirror_y].copy()
    x_local_crystal_near[mirror_y] =  x_local_crystal_far[mirror_y]
    del x_local_crystal_far

    mirror_x = x_crystal_near > x_crystal_far
    x_crystal_near[mirror_x] = 127 - x_crystal_near[mirror_x]
    x_crystal_far[mirror_x] = 127 - x_crystal_far[mirror_x]
    x_local_crystal_near[mirror_x] = 7 - x_local_crystal_near[mirror_x]

    if (x_local_crystal_near >= slor_shape[1]).any():
        raise ValueError("x_local_crystal_near out of range")
    slors = (slors + x_local_crystal_near) * slor_shape[2]
    del x_local_crystal_near

    x_crystal_diff = x_crystal_far - x_crystal_near
    del x_crystal_far, x_crystal_near

    if (x_crystal_diff >= slor_shape[2]).any():
        raise ValueError("x_crystal_diff out of range")
    slors = (slors + x_crystal_diff) * slor_shape[3]
    del x_crystal_diff

    if (y_crystal_near >= slor_shape[3]).any():
        raise ValueError("y_crystal_near out of range")
    slors = (slors + y_crystal_near) * slor_shape[4]
    del y_crystal_near

    if (y_crystal_far >= slor_shape[4]).any():
        raise ValueError("y_crystal_far out of range")
    slors += y_crystal_far
    del y_crystal_far
    return slors


def lor_to_slor_bins(lors, slor_shape=None, system_shape=None):
    '''
    Converts LORs to SLORs using lor_to_slor and then bins them using
    numpy.bincount.

    Returns an array, shape = (np.prod(slor_shape),), representing the number
    of LORs with that SLOR index.

    default_slor_shape is used if system_shape is None.

    default_system_shape is used if system_shape is None.
    '''
    if slor_shape is None:
        slor_shape = default_slor_shape
    if system_shape is None:
        system_shape = default_system_shape

    slors = lor_to_slor(lors, slor_shape, system_shape)
    bins = np.bincount(slors, minlength=np.prod(slor_shape))
    return bins
