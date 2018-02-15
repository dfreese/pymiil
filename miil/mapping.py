#!/usr/bin/env python

import warnings
import numpy as np
from miil.defaults import (default_system_shape, default_system_shape_pcdrm)


def _check_pcdrmax_shape(shape):
    """

    """
    if len(shape) != 7:
        raise RuntimeError('(%s) has invalid length for PCDRMAX shape' % ','.join((str(x) for x in shape)))


def no_panels(system_shape=None):
    """
    Returns the number of panels for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return system_shape[0]


def no_cartridges(system_shape=None):
    """
    Returns the number of cartridges for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[0:2])


def no_fins(system_shape=None):
    """
    Returns the number of fins for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[0:3])


def no_modules(system_shape=None):
    """
    Returns the number of modules for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[0:4])


def no_apds(system_shape=None):
    """
    Returns the number of apds for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[0:5])


def no_crystals(system_shape=None):
    """
    Returns the number of crystals for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape)


def no_daqs(system_shape_pcdrm=None):
    """
    Returns the number of daq (4-up) boards for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[0:3])


def no_renas(system_shape_pcdrm=None):
    """
    Returns the number of renas for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[0:4])


def no_cartridges_per_panel(system_shape=None):
    """
    Returns the number of cartridges per panel for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[1:2])


def no_fins_per_panel(system_shape=None):
    """
    Returns the number of fins per panel for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[1:3])


def no_modules_per_panel(system_shape=None):
    """
    Returns the number of modules per panel for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[1:4])


def no_apds_per_panel(system_shape=None):
    """
    Returns the number of apds per panel for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[1:5])


def no_crystals_per_panel(system_shape=None):
    """
    Returns the number of crystals per panel for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[1:])


def no_daqs_per_panel(system_shape_pcdrm=None):
    """
    Returns the number of daq (4-up) boards per panel for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[1:3])


def no_renas_per_panel(system_shape_pcdrm=None):
    """
    Returns the number of renas per panel for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[1:4])


def no_fins_per_cartridge(system_shape=None):
    """
    Returns the number of fins per cartridge for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[2:3])


def no_modules_per_cartridge(system_shape=None):
    """
    Returns the number of modules per cartridge for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[2:4])


def no_apds_per_cartridge(system_shape=None):
    """
    Returns the number of apds per cartridge for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[2:5])


def no_crystals_per_cartridge(system_shape=None):
    """
    Returns the number of crystals per cartridge for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[2:])


def no_daqs_per_cartridge(system_shape_pcdrm=None):
    """
    Returns the number of daq (4-up) boards per cartridge for a given system
    shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[2:3])


def no_renas_per_cartridge(system_shape_pcdrm=None):
    """
    Returns the number of renas per cartridge for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[2:4])


def no_renas_per_daq(system_shape_pcdrm=None):
    """
    Returns the number of renas per daq board (4-up) for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[3:4])


def no_modules_per_daq(system_shape_pcdrm=None):
    """
    Returns the number of modules per daq (4-up) board for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[3:5])


def no_apds_per_daq(system_shape_pcdrm=None):
    """
    Returns the number of apds per daq (4-up) board for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[3:6])


def no_crystals_per_daq(system_shape_pcdrm=None):
    """
    Returns the number of crystlas per daq (4-up) board for a given system
    shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[3:7])


def no_modules_per_rena(system_shape_pcdrm=None):
    """
    Returns the number of modules per rena for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[4:5])


def no_apds_per_rena(system_shape_pcdrm=None):
    """
    Returns the number of apds per rena for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[4:6])


def no_crystals_per_rena(system_shape_pcdrm=None):
    """
    Returns the number of crystals per rena for a given system shape.
    default_system_shape_pcdrm is used if system_shape_pcdrm is None.

    """
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm
    _check_pcdrmax_shape(system_shape_pcdrm)
    return np.prod(system_shape_pcdrm[4:7])


def no_modules_per_fin(system_shape=None):
    """
    Returns the number of modules per fin for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[3:4])


def no_apds_per_fin(system_shape=None):
    """
    Returns the number of apds per fin for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[3:5])


def no_crystals_per_fin(system_shape=None):
    """
    Returns the number of crystals per fin for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[3:])


def no_apds_per_module(system_shape=None):
    """
    Returns the number of apds per module for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[4:5])


def no_crystals_per_module(system_shape=None):
    """
    Returns the number of crystals per module for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(system_shape[4:])


def no_crystals_per_apd(system_shape=None):
    """
    Returns the number of crystals per apd for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return system_shape[5]


def no_lors(system_shape=None):
    """
    Returns the number of LORs for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return no_crystals_per_panel(system_shape) ** 2


def slor_shape(system_shape=None):
    """
    Returns the SLOR shape for a given system shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    crystal_array_dim = int(np.sqrt(no_crystals_per_apd(system_shape)))
    shape = [
        no_fins_per_panel(system_shape),
        crystal_array_dim,
        crystal_array_dim * no_modules_per_fin(system_shape),
        crystal_array_dim * no_apds_per_module(system_shape),
        crystal_array_dim * no_apds_per_module(system_shape)]
    return shape


def no_slors(system_shape=None):
    """
    Returns the number of SLORs for a given SLOR shape.
    default_system_shape is used if system_shape is None.

    """
    if system_shape is None:
        system_shape = default_system_shape
    return np.prod(slor_shape(system_shape))

# For Calibrated Events


def cal_to_cartridge(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global catridge number for each
    event given system_shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge = events['cartridge'].astype(int) + \
        no_cartridges_per_panel(system_shape) * events['panel'].astype(int)
    return global_cartridge


def get_global_cartridge_number(events, system_shape=None):
    warnings.warn('miil.get_global_cartridge_number() changed to miil.cal_to_cartridge()',
                  DeprecationWarning, stacklevel=2)
    return cal_to_cartridge(events, system_shape=system_shape)


def cal_to_fin(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global fin number for each
    event given system_shape.  Uses cal_to_cartridge as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge = cal_to_cartridge(events, system_shape)
    global_fin = events['fin'] + no_fins_per_cartridge(system_shape) * global_cartridge
    return global_fin


def get_global_fin_number(events, system_shape=None):
    warnings.warn('miil.get_global_fin_number() changed to miil.cal_to_fin()',
                  DeprecationWarning, stacklevel=2)
    return cal_to_fin(events, system_shape=system_shape)


def cal_to_module(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global module number for each
    event given system_shape.  Uses cal_to_fin as a base.

    default_system_shape is used if system_shape is None.
    '''
    global_fin = get_global_fin_number(events, system_shape)
    global_module = events['module'] + no_modules_per_fin(system_shape) * global_fin
    return global_module


def get_global_module_number(events, system_shape=None):
    warnings.warn('miil.get_global_module_number() changed to miil.cal_to_module()',
                  DeprecationWarning, stacklevel=2)
    return cal_to_module(events, system_shape=system_shape)


def cal_to_apd(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global apd number for each
    event given system_shape.  Uses cal_to_module as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_module = cal_to_module(events, system_shape)
    global_apd = events['apd'] + no_apds_per_module(system_shape) * global_module
    return global_apd


def get_global_apd_number(events, system_shape=None):
    warnings.warn('miil.get_global_apd_number() changed to miil.cal_to_apd()',
                  DeprecationWarning, stacklevel=2)
    return cal_to_apd(events, system_shape=system_shape)


def cal_to_crystal(events, system_shape=None):
    '''
    Takes eventcal_dtype events and returns the global crystal number for each
    event given system_shape.  Uses cal_to_apd as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_apd = cal_to_apd(events, system_shape)
    global_crystal = events['crystal'] + no_crystals_per_apd(system_shape) * global_apd
    return global_crystal


def get_global_crystal_number(events, system_shape=None):
    warnings.warn('miil.get_global_crystal_number() changed to miil.cal_to_crystal()',
                  DeprecationWarning, stacklevel=2)
    return cal_to_crystal(events, system_shape=system_shape)

# For Coincidence Events


def coinc_to_cartridges(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    cartridge number for each event given system_shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge0 = events['cartridge0'].astype(int)
    global_cartridge1 = events['cartridge1'].astype(int) + \
        no_cartridges_per_panel(system_shape)
    return global_cartridge0, global_cartridge1


def get_global_cartridge_numbers(events, system_shape=None):
    warnings.warn('miil.get_global_cartridge_numbers() changed to miil.coinc_to_cartridges()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_cartridges(events, system_shape=system_shape)


def coinc_to_fins(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    fin number for each event given system_shape.  Uses
    coinc_to_cartridges as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_cartridge0, global_cartridge1 = \
        get_global_cartridge_numbers(events, system_shape)
    global_fin0 = events['fin0'] + no_fins_per_cartridge(system_shape) * global_cartridge0
    global_fin1 = events['fin1'] + no_fins_per_cartridge(system_shape) * global_cartridge1
    return global_fin0, global_fin1


def get_global_fin_numbers(events, system_shape=None):
    warnings.warn('miil.get_global_fin_numbers() changed to miil.coinc_to_fins()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_fins(events, system_shape=system_shape)


def coinc_to_modules(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    module number for each event given system_shape.  Uses
    coinc_to_fins as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_fin0, global_fin1 = get_global_fin_numbers(events, system_shape)
    global_module0 = events['module0'] + no_modules_per_fin(system_shape) * global_fin0
    global_module1 = events['module1'] + no_modules_per_fin(system_shape) * global_fin1
    return global_module0, global_module1


def get_global_module_numbers(events, system_shape=None):
    warnings.warn('miil.get_global_module_numbers() changed to miil.coinc_to_modules()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_modules(events, system_shape=system_shape)


def coinc_to_apds(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    apd number for each event given system_shape.  Uses
    coinc_to_modules as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_module0, global_module1 = \
        get_global_module_numbers(events, system_shape)
    global_apd0 = events['apd0'] + no_apds_per_module(system_shape) * global_module0
    global_apd1 = events['apd1'] + no_apds_per_module(system_shape) * global_module1
    return global_apd0, global_apd1


def get_global_apd_numbers(events, system_shape=None):
    warnings.warn('miil.get_global_apd_numbers() changed to miil.coinc_to_apds()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_apds(events, system_shape=system_shape)


def coinc_to_crystals(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the left and right global
    crystal number for each event given system_shape.  Uses
    get_global_apd_numbers as a base.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_apd0, global_apd1 = get_global_apd_numbers(events, system_shape)
    global_crystal0 = events['crystal0'] + no_crystals_per_apd(system_shape) * global_apd0
    global_crystal1 = events['crystal1'] + no_crystals_per_apd(system_shape) * global_apd1
    return global_crystal0, global_crystal1


def get_global_crystal_numbers(events, system_shape=None):
    warnings.warn('miil.get_global_crystal_numbers() changed to miil.coinc_to_crystals()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_crystals(events, system_shape=system_shape)


def coinc_to_lor(events, system_shape=None):
    '''
    Takes eventcoinc_dtype events and returns the global lor number for each
    event given system_shape.  Uses coinc_to_crystals as a base.
    Global lor calculated as:
        (global_crystal0 * no_crystals_per_panel) +
        (global_crystal1 - no_crystals_per_panel)

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_crystal0, global_crystal1 = \
        get_global_crystal_numbers(events, system_shape)

    return (global_crystal0 * no_crystals_per_panel(system_shape)) + \
           (global_crystal1 - no_crystals_per_panel(system_shape))


def get_global_lor_number(events, system_shape=None):
    warnings.warn('miil.get_global_lor_number() changed to miil.coinc_to_lor()',
                  DeprecationWarning, stacklevel=2)
    return coinc_to_lor(events, system_shape=system_shape)


def lor_to_crystals(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global crystal
    number based on the given system shape.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0 = lors // no_crystals_per_panel(system_shape)
    crystal1 = lors % no_crystals_per_panel(system_shape) + \
               no_crystals_per_panel(system_shape)
    return crystal0, crystal1


def get_crystals_from_lor(lors, system_shape=None):
    warnings.warn('miil.get_crystals_from_lor() changed to miil.lor_to_crystals()',
                  DeprecationWarning, stacklevel=2)
    return lor_to_crystals(lors, system_shape=system_shape)


def lor_to_apds(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global apd
    number based on the given system shape.  Uses lor_to_crystals as a base for
    calculation.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0, crystal1 = lor_to_crystals(lors, system_shape)
    apd0 = crystal0 // no_crystals_per_apd(system_shape)
    apd1 = crystal1 // no_crystals_per_apd(system_shape)
    return apd0, apd1


def get_apds_from_lor(lors, system_shape=None):
    warnings.warn('miil.get_apds_from_lor() changed to miil.lor_to_apds()',
                  DeprecationWarning, stacklevel=2)
    return lor_to_apds(lors, system_shape=system_shape)


def lor_to_modules(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global module
    number based on the given system shape.  Uses lor_to_crystals as a base for
    calculation.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0, crystal1 = lor_to_crystals(lors, system_shape)
    module0 = crystal0 // no_crystals_per_module(system_shape)
    module1 = crystal1 // no_crystals_per_module(system_shape)
    return module0, module1


def get_modules_from_lor(lors, system_shape=None):
    warnings.warn('miil.get_modules_from_lor() changed to miil.lor_to_modules()',
                  DeprecationWarning, stacklevel=2)
    return lor_to_modules(lors, system_shape=system_shape)


def lor_to_fins(lors, system_shape=None):
    '''
    Takes an array of lor indices and returns the left and right global fin
    numbers based on the given system shape.  Uses lor_to_crystals as a base
    for calculation.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0, crystal1 = lor_to_crystals(lors, system_shape)
    fin0 = crystal0 // no_crystals_per_fin(system_shape)
    fin1 = crystal1 // no_crystals_per_fin(system_shape)
    return fin0, fin1


def crystals_to_lor(crystal0, crystal1, system_shape=None, local_id=False):
    '''
    Converts two crystal indices to an lor number.  If local_id is False
    (default), it assumes the crystals are globablly indexed.  If True, it
    assumes they're indexed local to the panel.

    Lors are calculated as follows:
        lor = panel_crystal0 * no_crystals_per_panel + panel_crystal1

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0 = np.asarray(crystal0)
    crystal1 = np.asarray(crystal1)
    if not local_id:
        crystal1 -= no_crystals_per_panel(system_shape)
    lor = crystal0 * no_crystals_per_panel(system_shape) + crystal1
    return lor


def lor_to_slor(lors, system_shape=None):
    '''
    Breaks down an array of LOR indices and transforms them into SLOR indices.
    SLORs, or symmetric LORs, is a way of indicating LORs that see the same
    attenuation and are reflectively, rotationally, translationally symmetric,
    assuming a symmetric source.

    Slors are effecitvely addressed by their place in a five dimensional array.
    [fin_diff][near_x][x_diff][near_y][far_y].

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    slorshape = slor_shape(system_shape)

    crystal0 = lors // no_crystals_per_panel(system_shape)
    crystal1 = lors % no_crystals_per_panel(system_shape)

    fin0 = crystal0 // no_crystals_per_fin(system_shape)
    fin1 = crystal1 // no_crystals_per_fin(system_shape)
    fin_diff = np.abs(fin0 - fin1)
    del fin0, fin1
    if (fin_diff >= slorshape[0]).any():
        raise ValueError("fin_diff out of range")
    slors = fin_diff * slorshape[1]
    del fin_diff

    apd0 = (crystal0 // no_crystals_per_apd(system_shape)) % \
           no_apds_per_module(system_shape)
    apd1 = (crystal1 // no_crystals_per_apd(system_shape)) % \
           no_apds_per_module(system_shape)

    y_crystal_near = 7 - (crystal0 % 8) + 8 * apd0
    y_crystal_far = 7 - (crystal1 % 8) + 8 * apd1
    del apd0, apd1

    x_local_crystal_near = (crystal0 % no_crystals_per_apd(system_shape)) // 8
    x_local_crystal_far = 7 - (crystal1 % no_crystals_per_apd(system_shape)) // 8

    module0 = (crystal0 // no_crystals_per_module(system_shape)) % \
              no_modules_per_fin(system_shape)
    module1 = (crystal1 // no_crystals_per_module(system_shape)) % \
              no_modules_per_fin(system_shape)
    x_crystal_near = 8 * module0 + x_local_crystal_near
    x_crystal_far = 8 * module1 + x_local_crystal_far
    del module0, module1

    mirror_y = y_crystal_near > y_crystal_far
    y_crystal_near[mirror_y], y_crystal_far[mirror_y] = \
            y_crystal_far[mirror_y], y_crystal_near[mirror_y].copy()
    x_crystal_near[mirror_y], x_crystal_far[mirror_y] = \
            x_crystal_far[mirror_y], x_crystal_near[mirror_y].copy()
    x_local_crystal_near[mirror_y] = x_local_crystal_far[mirror_y]
    del x_local_crystal_far

    mirror_x = x_crystal_near > x_crystal_far
    x_crystal_near[mirror_x] = 127 - x_crystal_near[mirror_x]
    x_crystal_far[mirror_x] = 127 - x_crystal_far[mirror_x]
    x_local_crystal_near[mirror_x] = 7 - x_local_crystal_near[mirror_x]

    if (x_local_crystal_near >= slorshape[1]).any():
        raise ValueError("x_local_crystal_near out of range")
    slors = (slors + x_local_crystal_near) * slorshape[2]
    del x_local_crystal_near

    x_crystal_diff = x_crystal_far - x_crystal_near
    del x_crystal_far, x_crystal_near

    if (x_crystal_diff >= slorshape[2]).any():
        raise ValueError("x_crystal_diff out of range")
    slors = (slors + x_crystal_diff) * slorshape[3]
    del x_crystal_diff

    if (y_crystal_near >= slorshape[3]).any():
        raise ValueError("y_crystal_near out of range")
    slors = (slors + y_crystal_near) * slorshape[4]
    del y_crystal_near

    if (y_crystal_far >= slorshape[4]).any():
        raise ValueError("y_crystal_far out of range")
    slors += y_crystal_far
    del y_crystal_far
    return slors


def lor_to_slor_bins(lors, system_shape=None):
    '''
    Converts LORs to SLORs using lor_to_slor and then bins them using
    numpy.bincount.

    Returns an array, shape = (miil.no_slors(system_shape),), representing the
    number of LORs with that SLOR index.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape

    slors = lor_to_slor(lors, system_shape)
    bins = np.bincount(slors, minlength=no_slors(system_shape))
    return bins

def valid_slors(system_shape=None, keepdims=False):
    '''
    Returns a boolean mask based on the SLOR Shape that declares which SLOR bins
    can have an LOR assigned to them, as some are invalid, but the full array
    structure is maintained for simplicity.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    slorshape = slor_shape(system_shape)

    # The near y crystal, axis=3, must always be less than
    near_far_condition = (
        np.arange(slorshape[3])[None, None, None, :, None] <=
        np.arange(slorshape[4])[None, None, None, None, :])

    # The sum of the x local near crystal, axis=1, and the x difference,
    # axis=2, must be less than the full width of the panel, which is
    # represented by the size of the x difference dimension, axis=2.
    panel_width_condition = (
        (np.arange(slorshape[2])[None, None, :, None, None] +
         np.arange(slorshape[1])[None, :, None, None, None]) < slorshape[2])

    valids = np.ones(slorshape, dtype=bool)
    valids &= near_far_condition
    valids &= panel_width_condition

    if keepdims:
        return valids
    else:
        return valids.ravel()

def check_pcfmax(
        panel=None, cartridge=None, fin=None, module=None,
        apd=None, crystal=None, system_shape=None):
    '''
    Checks if the numbers are valid given the system shape
    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape

    valid = True
    if panel is not None:
        panel = np.asarray(panel)
        valid |= (panel >= 0).all()
        valid |= (panel < no_panels(system_shape)).all()
    if cartridge is not None:
        cartridge = np.asarray(cartridge)
        valid |= (cartridge >= 0).all()
        valid |= (cartridge < no_cartridges_per_panel(system_shape)).all()
    if fin is not None:
        fin = np.asarray(fin)
        valid |= (fin >= 0).all()
        valid |= (fin < no_fins_per_cartridge(system_shape)).all()
    if module is not None:
        module = np.asarray(module)
        valid |= (module >= 0).all()
        valid |= (module < no_modules_per_fin(system_shape)).all()
    if apd is not None:
        apd = np.asarray(apd)
        valid |= (apd >= 0).all()
        valid |= (apd < no_apds_per_module(system_shape)).all()
    if crystal is not None:
        crystal = np.asarray(crystal)
        valid |= (crystal >= 0).all()
        valid |= (crystal < no_crystals_per_apd(system_shape)).all()

    return valid


def pcfm_to_pcdrm(
        panel, cartridge, fin, module,
        system_shape_pcfm=None, system_shape_pcdrm=None):
    """

    """

    if system_shape_pcfm is None:
        system_shape_pcfm = default_system_shape
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm

    if not check_pcfmax(panel, cartridge, fin, module):
        raise RuntimeError('Invalid values for system shape')

    modules_per_rena = no_modules_per_rena(system_shape_pcdrm)
    renas_per_daq = no_renas_per_daq(system_shape_pcdrm)

    rena = (
        2 * ((no_fins_per_cartridge(system_shape_pcfm) - 1 - fin) // 2) +
        1 * ((panel == 0) & (module % 8 >= modules_per_rena)) +
        1 * ((panel == 1) & (module % 8 < modules_per_rena))
        )
    rena_local_module = (
        (module % modules_per_rena) * (panel == 0) +
        (modules_per_rena - 1 - (module % modules_per_rena)) * (panel == 1)
        )
    daq = (
        2 * ((panel == 0) & (fin % 2) & (renas_per_daq > 2)) +
        1 * ((panel == 0) & (module >= 8)) +
        2 * ((panel == 1) & ((fin % 2) == 0) & (renas_per_daq > 2)) +
        1 * ((panel == 1) & (module < 8))
        )
    return daq, rena, rena_local_module

def pcdrm_to_pcfm(
        panel, cartridge, daq, rena, rena_local_module,
        system_shape_pcfm=None, system_shape_pcdrm=None):
    """

    """

    if system_shape_pcfm is None:
        system_shape_pcfm = default_system_shape
    if system_shape_pcdrm is None:
        system_shape_pcdrm = default_system_shape_pcdrm

    module = (
        rena_local_module +
        no_modules_per_rena(system_shape_pcdrm) * (rena % 2) +
        no_modules_per_fin(system_shape_pcfm) / 2 * (daq % 2)
        )
    module[panel == 1] = (
        no_modules_per_fin(system_shape_pcfm) - 1 - module[panel == 1]
        )

    renas_per_daq = no_renas_per_daq(system_shape_pcdrm)

    fin = (
        no_fins_per_cartridge(system_shape_pcfm) - 1 - 2 * (rena // 2) +
        -1 * ((panel == 0) & (daq < 2) & (renas_per_daq > 2)) +
        -1 * ((panel == 1) & (daq >= 2) & (renas_per_daq > 2))
        )
    return fin, module
