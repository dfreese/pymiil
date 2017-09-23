import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit

from miil.version import __version__
from miil.defaults import *
from miil.types import *
from miil.io import *
from miil.mapping import *
from miil.position import *
from miil.opt import mlem
from miil.plot import set_y_axis_order
from miil.normalization import (
    get_module_distribution, get_apd_distribution, get_crystal_distribution)
from miil.recon import BreastPETSystemMatrix

def tcal_coinc_events(
        events, tcal,
        system_shape=None,
        uv_period_ns=1024.41,
        breast_daq_json_config=None):
    '''
    Takes an array of eventcoinc_dtype events and applies a time calibration,
    making sure the dtf are then wrapped to the uv period in nanoseconds.  The
    default uv period represent 980kHz in nanoseconds.

    If tcal is a string, load_time_calibration will be called to load the time
    calibration.

    Returned array is a copy of the original array.

    default_system_shape is used if system_shape is None.
    '''
    if system_shape is None:
        system_shape = default_system_shape

    # If tcal is a string, load in that time calibration first.
    if isinstance(tcal, str):
        tcal = load_time_calibration(tcal)

    idx0, idx1 = get_global_crystal_numbers(events, system_shape)
    ft0_offset = tcal['offset'][idx0] + \
        tcal['edep_offset'][idx0] * (events['E0'] - 511)
    ft1_offset = tcal['offset'][idx1] + \
        tcal['edep_offset'][idx1] * (events['E1'] - 511)
    cal_events = events.copy()
    cal_events['dtf'] -= ft0_offset
    cal_events['dtf'] += ft1_offset

    # Load in the uv_period_ns from a json file if specified.
    if breast_daq_json_config is not None:
        uv_period_ns = load_uv_period(breast_daq_json_config) * 1e9

    # wrap all of the events to the specified uv period.
    while np.any(cal_events['dtf'] > uv_period_ns):
        cal_events['dtf'][cal_events['dtf'] > uv_period_ns] -= uv_period_ns
    while np.any(cal_events['dtf'] < -uv_period_ns):
        cal_events['dtf'][cal_events['dtf'] < -uv_period_ns] += uv_period_ns

    return cal_events


def create_listmode_data(
        events,
        system_shape=default_system_shape,
        panel_sep=default_panel_sep,
        list_type=0):
    '''
    Creates an array of list mode data in either cudarecon_type0_vec_dtype or
    cudarecon_type1_vec_dtype from eventcoinc data by calling get_crystal_pos.

    Parameters
    ----------
    events : (n,) shaped ndarray of eventcoinc_dtype
        Scalar or array of coincidence events

    Returns
    -------
    events : (n,) shaped ndarray of cudarecon_type[0,1]_vec_dtype
        Scalar or array of list mode data for cudarecon.
    '''
    dtype = cudarecon_type0_vec_dtype
    if list_type == 1:
        dtype = cudarecon_type1_vec_dtype

    lm_data = np.zeros(events.shape, dtype=dtype)
    lm_data['pos0'], lm_data['pos1'] = get_crystal_pos(
        events, system_shape=system_shape, panel_sep=panel_sep)
    return lm_data


def create_listmode_from_vec(
        vec, panel_sep=default_panel_sep,
        system_shape=default_system_shape):
    '''
    Creates an array of list mode data in cudarecon_type1_vec_dtype from
    a sparse column vector representing the counts on each lor.

    Parameters
    ----------
    vec : (n,1) csc_matrix
        sparse column matrix of lor counts

    Returns
    -------
    events : (n,) shaped ndarray of cudarecon_type1_vec_dtype
        Scalar or array of list mode data for cudarecon.
    '''
    lm_data = np.zeros((vec.nnz,), dtype=cudarecon_type1_vec_dtype)
    lm_data['pos0'], lm_data['pos1'] = get_lor_positions(vec.indices,
                                                         system_shape,
                                                         panel_sep)
    lm_data['weight'] = vec.data.copy()
    return lm_data


def create_listmode_from_lors(
        lors,
        panel_sep=default_panel_sep,
        system_shape=default_system_shape,
        list_type=0):
    '''
    Creates an array of list mode data in cudarecon_type[0,1]_vec_dtype from
    an array of lor indices.  Calls get_lor_positions.

    Parameters
    ----------
    vec : (n,1) csc_matrix
        sparse column matrix of lor counts
    list_type : scalar
        indicates cudarecon_type.  values 0 and 1 indicate
        cudarecon_type0_vec_dtype and cudarecon_type1_vec_dtype respectively.

    Returns
    -------
    events : (n,) shaped ndarray of cudarecon_type[0,1]_vec_dtype
        Scalar or array of list mode data for cudarecon.
    '''
    dtype = cudarecon_type0_vec_dtype
    if list_type == 1:
        dtype = cudarecon_type1_vec_dtype

    lm_data = np.zeros(lors.shape, dtype=dtype)
    lm_data['pos0'], lm_data['pos1'] = get_lor_positions(
        lors, system_shape, panel_sep)
    return lm_data


def correct_resets(data, threshold=1.0e3):
    '''
    Corrects any accidental resets in the coarse timestamp during a run of the
    system.

    Parameters
    ----------
    data : (n,) ndarray of eventraw_dtype or eventcal_dtype
        Array of raw or calibrated event data
    threshold : scalar
        The negative jump that constitues a reset in the coarse timestamp.
        Should be large enough as to not include events that might be out of
        order.

    Returns
    -------
    data : (n,) shaped ndarray of cudarecon_type[0,1]_vec_dtype
        data array with 'ct' values corrected for resets
    '''
    data['ct'][1:] = np.diff(data['ct'])
    # Assume that any negative should just be the next coarse timestampe tick,
    # so we set it to one, so that the ct is incremented in the cumsum step
    data['ct'][data['ct'] < -threshold] = 1
    data['ct'] = np.cumsum(data['ct'])
    return data


def save_sparse_csc(filename, array):
    '''
    Saves a csc_matrix to a file by writing out the individual 'data',
    'indices', and 'indptr' arrays to a numpy zip file.

    Parameters
    ----------
    filename : str
        String of the file name to which to write the npz file.
    array : csc_matrix
        Any scipy csc_matrix
    '''
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csc(filename):
    '''
    Loads a csc_matrix saved by save_sparse_csc.  Assumes filename is a npz
    file containing 'data', 'indices', and 'indptr' arrays.

    Parameters
    ----------
    filename : str
        String of the npz file containing the csc_matrix's arrays.

    Returns
    -------
    data : csc_matrix
        A csc_matrix created from the 'data', 'indices', and 'indptr' arrays.
    '''
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def create_sparse_column_vector(data, size=None):
    '''
    Creates a sparse column vector using a scipy.csc_matrix.  Takes data to be
    a count for that particular index.  Same indices will be summed in the
    column vector.  The vector will be (1,size) in shape.  If size is None,
    then the maximum index in data determines the size of the vector.  Used
    primarily for lor indices.

    Parameters
    ----------
    data : array
        A list of indices.

    Returns
    -------
    vec : (1, size) shaped csc_matrix
        A csc_matrix created from the indices in data.
    '''
    shape = None
    if size is not None:
        shape = (int(size), 1)
    data = np.asarray(data)
    data.sort()
    return csc_matrix((np.ones((len(data),), dtype=float),
                       (data.astype(int), np.zeros((len(data),), dtype=int))
                      ), shape=shape)

def gauss_function(x, a, mu, sigma, dc_offset=0.0):
    '''
    Evaluate a gaussian of mean mu, std of sigma, and amplitude a at a point x.
    Primarily used for fitting histograms.

    Parameters
    ----------
    x : ndarray or scalar
        Points to be evaluated
    a : scalar
        Amplitude of the gaussian
    mu : scalar
        Mean of the gaussian
    sigma : scalar
        Standard deviation of the gaussian
    dc_offset : scalar
        a baseline offset to add to the entire function

    Returns
    -------
    val : array or scalar
        gauss(x) = a * exp(-(x - mu)**2 / (2 * sigma**2))
    '''
    return a * np.exp(-(x - mu)**2.0 / (2 * sigma**2)) + dc_offset


def fit_hist_gauss(n, edges, p0=None, dc_offset=False):
    '''
    Takes the output of a histogram and fits a gaussian function to it.
    Scipy curve_fit is uses in combination with gauss_function to do a
    non-linear fit.  The fit is initialize to the mean and variance of the
    given data.

    Parameters
    ----------
    n : ndarray
        Array of bin values from histogramming a set of data.
    edges : ndarray
        Array of bin edges from histogramming a set of data.
    p0 : array
        Initial guess of parameters for the fit [a, mu, sigma]
    dc_offset : bool
        If a dc_offset should be added to the fit

    Returns
    -------
    popt : array
        Array with 3 elements [a, mu, sigma] of the optimal fit.
        Is 4 elements [a, mu, sigma, offset] if dc_offset is true.

    Examples
    --------
    >>> data = 1.0 + 2 * np.random.randn(1000)
    >>> n, edges, patches = plt.hist(data)
    >>> popt = miil.fit_hist_gauss(n, edges)
    '''
    # find the centers of the bins
    centers = (edges[1:] + edges[:-1]) / 2.0
    if p0 is None:
        # If we're not given an initial guess, then do a weighted average of
        # the centers to initialize the estimate of the mean
        mean = np.average(centers, weights=n)
        # Then do a weighted average to estimate the variance for sigma
        sigma = np.sqrt(np.average((centers - mean) ** 2, weights=n))
        offset = (n[0] + n[-1]) / 2.0
        if dc_offset:
            p0 = [np.max(n), mean, sigma, offset]
        else:
            p0 = [np.max(n), mean, sigma]
    bounds = ((0, -np.inf, 0), (np.inf, np.inf, np.inf))
    if dc_offset:
        bounds = ((0, -np.inf, 0, 0), (np.inf, np.inf, np.inf, np.max(n)))
    popt = curve_fit(gauss_function, centers, n, p0=p0, bounds=bounds)[0]
    return popt


def bootstrap_gauss_fit(data, n, bins, range=None, dc_offset=False):
    """
    Take points from data, split them into n groups randomly, then histogram
    them across range with bins number of bins.

    Parameters
    ----------
    data : ndarray
        Raw data points from a gaussian distribution to be fit.
    n : int
        Number of samples to bootstrap out of the original data
    bins : int, or ndarray
        Either the number of bins in a range or an array of bin edges for
        histogramming the data.
    range : tuple of scalars, shape (2,)
        A tuple of scalar values defining the edge of the histogram range.  If
        bins is not an integer, this value is ignored.  See numpy.histogram
    dc_offset : bool
        If a dc_offset should be added to the fit

    Returns
    -------
    popt_mean : array
        Array with 3 elements [a, mu, sigma] of the optimal fit.
        Is 4 elements [a, mu, sigma, offset] if dc_offset is true.
    popt_err : array
        Array with 3 elements [a_err, mu_err, sigma_err], which is the standard
        error of the optimal fit, calculated from the n bootstrapped samples.
        Is 4 elements [a_err, mu_err, sigma_err, offset_err] if dc_offset is
        true.

    Examples
    --------
    >>> data = 1.0 + 2 * np.random.randn(1000)
    >>> pm, pe = miil.bootstrap_gauss_fit(data, 10, 100, (-5, 5))

    """
    groups = np.random.randint(n, size=data.size)
    popts = np.zeros((n, 3))
    if dc_offset:
        popts = np.zeros((n, 4))
    popt = None

    for ii in xrange(n):
        counts, edges = np.histogram(data[groups == ii], bins=bins, range=range)
        popt = fit_hist_gauss(counts, edges, popt, dc_offset)
        popts[ii, :] = popt
    popt_mean = np.mean(popts, axis=0)
    popt_err = np.std(popts, axis=0, ddof=1) / np.sqrt(popts.shape[0])

    return popt_mean, popt_err

def eval_gauss_over_range(popt, n=100, range=None, edges=None):
    '''
    Evaluates a gaussian function over a range, traditionally from the output
    of fit_hist_gauss.

    Parameters
    ----------
    popt : array
        Array with 3 elements [a, mu, sigma] of the optimal fit.
    n : scalar
        number of points at which to evaluate the fit
    range : array shape=(2,)
        The min and max of the range over which to evaluate the fit.
    edges : ndarray
        Array of bin edges from histogramming a set of data.  Sets the range
        over which the fit is calculated. Equivalent to
        range=(edges.min(), edges.max()).

    Returns
    -------
    x : array
        points where the fit was evaluated
    y : array
        value of the fit where the it was evaluated

    Examples
    --------
    >>> data = 1.0 + 2 * np.random.randn(1000)
    >>> n, edges, patches = plt.hist(data)
    >>> popt = miil.fit_hist_gauss(n, edges)
    >>> x_fit, y_fit = miil.eval_gauss_over_range(popt, 200, edges=edges)
    >>> plt.plot(x_fit, y_fit)
    >>> plt.show()
    '''
    if edges is not None:
        range_min = edges.min()
        range_max = edges.max()
    if range is not None:
        range_min = range[0]
        range_max = range[1]
    x = np.linspace(range_min, range_max, n)
    y = gauss_function(x, *popt)
    return x, y


def main():
    return

if __name__ == '__main__':
    main()
