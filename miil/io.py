#!/usr/bin/env python

import os
import json
import numpy as np
from miil.types import *


def load_cuda_vox(filename, return_header=False):
    '''
    Takes a cuda vox image file and loads it into a (X,Y,Z) shaped numpy array.
    if return_header is True, then a header object is returned with a
    cuda_vox_file_header_dtype.

    Assumes image was written as [x][z][y] on disk as float32 values.
    '''
    fid = open(filename, 'rb')
    header = np.fromfile(fid, dtype=cuda_vox_file_header_dtype, count=1)[0]
    image = np.fromfile(fid, dtype=np.float32)
    fid.close()
    image = np.reshape(image, (header['size'][0],
                               header['size'][2], header['size'][1]))
    image = np.swapaxes(image, 1, 2)
    if return_header:
        return image, header
    else:
        return image


def write_cuda_vox(image, filename, magic_number=65531, version_number=1):
    '''
    Takes a cuda vox image in the form of a numpy array shaped (X,Y,Z) and
    writes it to disk in the cuda vox image header format (see
    cuda_vox_file_header_dtype).  Header defaults are based on projection
    defaults.

    Writes the image as [x][z][y] on the disk.  Values are written as
    float32.
    '''
    header = np.zeros((1,), dtype=cuda_vox_file_header_dtype)
    header['magic_number'] = magic_number
    header['version_number'] = version_number
    header['size'] = image.shape
    with open(filename, 'wb') as fid:
        header.tofile(fid)
        image.swapaxes(1, 2).astype(np.float32).tofile(fid)


def load_cuda_vox_shape(filename):
    '''
    Takes a cuda vox image file and loads the shape (X,Y,Z) of the image from
    the cuda vox header.

    '''
    fid = open(filename, 'rb')
    header = np.fromfile(fid, dtype=cuda_vox_file_header_dtype, count=1)[0]
    fid.close()
    image_shape = (header['size'][0], header['size'][1], header['size'][2])
    return image_shape


def load_amide(filename, size):
    '''
    Loads an amide image file of size (X, Y, Z).

    Assumes image was written as [z][y][x] on disk.  Amide format has no header
    information, so the image size must be known.  Assumes values are written
    as float32.
    '''
    return np.fromfile(filename,
                       dtype=np.float32).reshape(size[::-1]).swapaxes(0, 2)


def write_amide(image, filename):
    '''
    Writes an amide image file of size (X, Y, Z) in [z][y][x] order as float32.
    Amide format has no header information, so the image size must be known.
    '''
    image.swapaxes(0, 2).astype(np.float32).tofile(filename)


def load_decoded(filename, count=-1):
    '''
    Load a decode file.  This is a binary file of eventraw_dtype objects.  If
    count is -1 all events will be loaded, otherwise count events will be
    loaded.
    '''
    with open(filename, 'rb') as fid:
        data = np.fromfile(fid, dtype=eventraw_dtype, count=count)
    return data


def load_calibrated(filename, count=-1):
    '''
    Load a calibrate file.  This is a binary file of eventcal_dtype objects.
    If count is -1 all events will be loaded, otherwise count events will be
    loaded.
    '''
    with open(filename, 'rb') as fid:
        data = np.fromfile(fid, dtype=eventcal_dtype, count=count)
    return data


def load_coincidence(filename, count=-1):
    '''
    Load a calibrate file.  This is a binary file of eventcoinc_dtype objects.
    If count is -1 all events will be loaded, otherwise count events will be
    loaded.
    '''
    with open(filename, 'rb') as fid:
        data = np.fromfile(fid, dtype=eventcoinc_dtype, count=count)
    return data


def get_filenames_from_filelist(filename):
    '''
    Helper function to load in filelist.  Corrects file paths as relative to
    the filelist if the paths that are listed are not absolute.

    Takes the path of the filelist.

    Returns a list of corrected filenames.
    '''
    # Get all of the lines out of the file
    with open(filename, 'r') as fid:
        files = fid.read().splitlines()
    # Get the directory of the filelist
    filename_path = os.path.dirname(filename)

    # Assume each line in the coinc filelist is either an absolute directory or
    # referenced to the directory of the file.
    full_files = []
    for local_file in files:
        if os.path.isabs(local_file):
            full_files.append(local_file)
        else:
            if not filename_path:
                full_files.append(local_file)
            else:
                full_files.append(filename_path + '/' + local_file)
    # Now we have a list of files fully corrected relative to their filelist
    return full_files


def load_decoded_filelist(filename, count=-1):
    '''
    Call load_decoded for every file in the given filelist.  count decode
    events are loaded from each file in the filelist.
    '''
    files = get_filenames_from_filelist(filename)
    data = np.hstack([load_decoded(f, count) for f in files])
    return data


def load_calib_filelist(filename, count=-1):
    '''
    Call load_calibrated for every file in the given filelist.  count calibrate
    events are loaded from each file in the filelist.
    '''
    files = get_filenames_from_filelist(filename)
    data = np.hstack([load_calibrated(f, count) for f in files])
    return data


def load_coinc_filelist(filename, count=-1):
    '''
    Call load_coincidence for every file in the given filelist.  count coinc
    events are loaded from each file in the filelist.
    '''
    files = get_filenames_from_filelist(filename)
    data = np.hstack([load_coincidence(f, count) for f in files])
    return data


def load_pedestals(filename):
    '''
    Loads a pedestal file.  Should be a text file with space separated columns
    for each value in ped_dtype.
    '''
    return np.loadtxt(filename, dtype=ped_dtype)


def load_locations(filename):
    '''
    Loads a crystal location file (typically .loc).  Should be a text file with
    space separated columns for each value in loc_dtype.
    '''
    return np.loadtxt(filename, dtype=loc_dtype)


def write_locations(cal, filename):
    '''
    Writes a crystal location file (typically .loc) to filename.
    '''
    return np.savetxt(filename, cal, '%d %0.6f %0.6f')


def load_calibration(filename):
    '''
    Loads a crystal calibraiton file (typically .cal).  Should be a text file
    with space separated columns for each value in cal_dtype.
    '''
    return np.loadtxt(filename, dtype=cal_dtype)


def write_calibration(cal, filename):
    '''
    Writes a crystal calibration file (typically .cal) to filename.  Writes the
    following columns

    - use %d
    - x %0.6f
    - y %0.6f
    - gain_spat %0.0f
    - gain_comm %0.0f
    - eres_spat %0.4f
    - eres_comm %0.4f
    '''
    return np.savetxt(filename, cal, '%d %0.6f %0.6f %0.0f %0.0f %0.4f %0.4f')


def load_time_calibration(filename):
    '''
    Loads a crystal time calibraiton file (typically .tcal).  Should be a text
    file with space separated columns for each value in tcal_dtype.
    '''
    return np.loadtxt(filename, dtype=tcal_dtype)


def load_system_shape_pcfmax(filename):
    '''
    From a json system configuration file, load the PCFMAX shape of the system.
    Assumes that there are 2 apds per module and 64 crystals per apd, which is
    fixed in hardware.
    '''
    with open(filename, 'r') as fid:
        config = json.load(fid)
    system_config = config['system_config']
    system_shape = [system_config['NUM_PANEL_PER_DEVICE'],
                    system_config['NUM_CART_PER_PANEL'],
                    system_config['NUM_FIN_PER_CARTRIDGE'],
                    system_config['NUM_MODULE_PER_FIN'],
                    2, 64, ]
    return system_shape


def load_uv_freq(filename):
    '''
    From a json system configuration file, load the uv_frequency.
    '''
    with open(filename, 'r') as fid:
        config = json.load(fid)
    return config['uv_frequency']


def load_uv_period(filename):
    '''
    Use load_uv_freq to calculate the uv period from a json system
    configuration file.
    '''
    uv_freq = load_uv_freq(filename)
    return 1.0 / uv_freq

def load_lors(filename):
    '''
    Load an lor list file.  Primarily a convinience function to not have to
    remember int64.
    '''
    return np.fromfile(filename, dtype=np.int64)
