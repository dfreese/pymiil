#!/usr/bin/env python

import numpy as np
from miil.defaults import (
    default_system_shape, default_panel_sep, default_x_crystal_pitch,
    default_y_crystal_pitch, default_x_module_pitch,
    default_y_apd_pitch, default_y_apd_offset, default_z_pitch)
from miil.mapping import (
    get_crystals_from_lor, check_pcfmax, no_crystals,
    no_cartridges_per_panel, no_fins_per_cartridge, no_modules_per_fin,
    no_apds_per_module, no_crystals_per_apd,
    no_crystals_per_panel, no_crystals_per_cartridge,
    no_crystals_per_fin, no_crystals_per_module)


def force_array(x, dtype=None):
    '''
    Helper function to force a value to be represented as a numpy array whether
    or not it is a scalar, list, or numpy array type.
    '''
    if np.isscalar(x):
        x = (x,)
    return np.asarray(x, dtype=dtype)


def get_position_pcfmax(
        panel, cartridge, fin, module, apd, crystal,
        system_shape=None,
        panel_sep=default_panel_sep,
        x_crystal_pitch=default_x_crystal_pitch,
        y_crystal_pitch=default_y_crystal_pitch,
        x_module_pitch=default_x_module_pitch,
        y_apd_pitch=default_y_apd_pitch,
        y_apd_offset=default_y_apd_offset,
        z_pitch=default_z_pitch):
    '''
    Calculates the position of a crystal based upon it's PCFMAX number.
    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    index_valid = check_pcfmax(
        panel, cartridge, fin, module, apd, crystal, system_shape)
    if not index_valid:
        raise ValueError('one or more values out of bounds')

    panel = force_array(panel, dtype=float)
    cartridge = force_array(cartridge, dtype=float)
    fin = force_array(fin, dtype=float)
    module = force_array(module, dtype=float)
    apd = force_array(apd, dtype=float)
    crystal = force_array(crystal, dtype=float)

    positions = np.zeros((len(panel), 3), dtype=float)

    positions[:, 0] = (x_module_pitch - 8 * x_crystal_pitch) / 2 + \
        (module - 8) * x_module_pitch

    positions[panel == 0, 0] += x_crystal_pitch * \
        ((crystal[panel == 0] // 8) + 0.5)

    positions[panel == 1, 0] += x_crystal_pitch * \
        (7 - (crystal[panel == 1] // 8) +
         0.5)

    positions[:, 1] = panel_sep / 2.0 + y_apd_offset + \
        apd * y_apd_pitch + \
        (7 - crystal % 8 + 0.5) * y_crystal_pitch

    positions[:, 2] = z_pitch * (
        0.5 + fin + no_fins_per_cartridge(system_shape) * (
            cartridge - no_cartridges_per_panel(system_shape) / 2.0))

    positions[panel == 0, 1] *= -1

    return positions


def get_position_global_crystal(
        global_crystal_ids,
        system_shape=None,
        panel_sep=default_panel_sep,
        x_crystal_pitch=default_x_crystal_pitch,
        y_crystal_pitch=default_y_crystal_pitch,
        x_module_pitch=default_x_module_pitch,
        y_apd_pitch=default_y_apd_pitch,
        y_apd_offset=default_y_apd_offset,
        z_pitch=default_z_pitch):
    '''
    Get the crystal position based on the global crystal id number.  Does this
    by calling get_position_pcfmax.

    Parameters
    ----------
    global_crystal_ids : scalar or (n,) shaped ndarray
        Scalar or array of globabl crystal ids
    system_shape : list like
        List or array describing the shape of the system.
        miil.default_system_shape is used if it is None.

    Returns
    -------
    p : (n,3) array
        x, y, z positions of the crystals.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    global_crystal_ids = force_array(global_crystal_ids, dtype=float)

    if np.any(global_crystal_ids >= no_crystals(system_shape)):
        raise ValueError('One or more crystal ids are out of range')
    elif np.any(global_crystal_ids < 0):
        raise ValueError('One or more crystal ids are out of range')

    panel = global_crystal_ids // no_crystals_per_panel(system_shape)
    cartridge = (global_crystal_ids //
                 no_crystals_per_cartridge(system_shape)) % \
                    no_cartridges_per_panel(system_shape)
    fin = (global_crystal_ids // no_crystals_per_fin(system_shape)) % \
            no_fins_per_cartridge(system_shape)
    module = (global_crystal_ids // no_crystals_per_module(system_shape)) % \
            no_modules_per_fin(system_shape)
    apd = (global_crystal_ids // no_crystals_per_apd(system_shape)) % \
            no_apds_per_module(system_shape)
    crystal = global_crystal_ids % no_crystals_per_apd(system_shape)

    return get_position_pcfmax(panel, cartridge, fin, module, apd, crystal,
                               system_shape, panel_sep, x_crystal_pitch,
                               y_crystal_pitch, x_module_pitch, y_apd_pitch,
                               y_apd_offset, z_pitch)


def get_positions_cal(
        events,
        system_shape=None,
        panel_sep=default_panel_sep,
        x_crystal_pitch=default_x_crystal_pitch,
        y_crystal_pitch=default_y_crystal_pitch,
        x_module_pitch=default_x_module_pitch,
        y_apd_pitch=default_y_apd_pitch,
        y_apd_offset=default_y_apd_offset,
        z_pitch=default_z_pitch):
    '''
    Get the crystal position based on the eventcal_dtype event.  Does this
    by calling get_position_pcfmax.

    Parameters
    ----------
    events : (n,) shaped ndarray of eventcal_dtype
        Scalar or array of calibrated events
    system_shape : list like
        List or array describing the shape of the system.
        miil.default_system_shape is used if it is None.

    Returns
    -------
    p : (n,3) array
        x, y, z positions of the crystals.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    pos = get_position_pcfmax(
        events['panel'], events['cartridge'], events['fin'],
        events['module'], events['apd'], events['crystal'],
        system_shape, panel_sep, x_crystal_pitch, y_crystal_pitch,
        x_module_pitch, y_apd_pitch, y_apd_offset, z_pitch)
    return pos


def get_crystal_pos(
        events,
        system_shape=None,
        panel_sep=default_panel_sep,
        x_crystal_pitch=default_x_crystal_pitch,
        y_crystal_pitch=default_y_crystal_pitch,
        x_module_pitch=default_x_module_pitch,
        y_apd_pitch=default_y_apd_pitch,
        y_apd_offset=default_y_apd_offset,
        z_pitch=default_z_pitch):
    '''
    Get the left and right crystal position based on the eventcoinc_dtype
    event. Does this by calling get_position_pcfmax.

    Parameters
    ----------
    events : (n,) shaped ndarray of eventcoinc_dtype
        Scalar or array of coincidence events
    system_shape : list like
        List or array describing the shape of the system.
        miil.default_system_shape is used if it is None.

    Returns
    -------
    p0 : (n,3) array
        x, y, z positions of the left crystals.
    p1 : (n,3) array
        x, y, z positions of the right crystals.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    pos0 = get_position_pcfmax(
        np.zeros(events.shape), events['cartridge0'], events['fin0'],
        events['module0'], events['apd0'], events['crystal0'],
        system_shape, panel_sep, x_crystal_pitch, y_crystal_pitch,
        x_module_pitch, y_apd_pitch, y_apd_offset, z_pitch)

    pos1 = get_position_pcfmax(
        np.ones(events.shape), events['cartridge1'], events['fin1'],
        events['module1'], events['apd1'], events['crystal1'],
        system_shape, panel_sep, x_crystal_pitch, y_crystal_pitch,
        x_module_pitch, y_apd_pitch, y_apd_offset, z_pitch)

    return pos0, pos1


def get_lor_positions(
        lors,
        system_shape=None,
        panel_sep=default_panel_sep,
        x_crystal_pitch=default_x_crystal_pitch,
        y_crystal_pitch=default_y_crystal_pitch,
        x_module_pitch=default_x_module_pitch,
        y_apd_pitch=default_y_apd_pitch,
        y_apd_offset=default_y_apd_offset,
        z_pitch=default_z_pitch):
    '''
    Get the left and right crystal position based on the lor index.
    Does this by calling get_position_pcfmax.

    Parameters
    ----------
    lors : (n,) shaped ndarray
        Scalar or array of lor indices
    system_shape : list like
        List or array describing the shape of the system.
        miil.default_system_shape is used if it is None.

    Returns
    -------
    p0 : (n,3) array
        x, y, z positions of the left crystals.
    p1 : (n,3) array
        x, y, z positions of the right crystals.
    '''
    if system_shape is None:
        system_shape = default_system_shape
    crystal0, crystal1 = get_crystals_from_lor(lors, system_shape)
    line_start = get_position_global_crystal(
        crystal0, system_shape, panel_sep, x_crystal_pitch, y_crystal_pitch,
        x_module_pitch, y_apd_pitch, y_apd_offset, z_pitch)
    line_end = get_position_global_crystal(
        crystal1, system_shape, panel_sep, x_crystal_pitch, y_crystal_pitch,
        x_module_pitch, y_apd_pitch, y_apd_offset, z_pitch)
    return line_start, line_end
