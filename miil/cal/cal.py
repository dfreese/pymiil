import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import lsmr, lsqr
import miil

def cal_time(data, system_shape=None, edep_type='linear',
        xtal_thres=None, apd_thres=None, admm_iter=5, admm_rho=None,
        admm_invrho=None, admm_fwhm_exp=16.0, xtal_global=False):
    """
    Implementation of the method in "Robust Timing Calibration for PET Using
    L1-Norm Minimization" by Freese, Hsu, Innes, and Levin, 2017.  This
    implementation is specifically for the Breast Panel system.

    Parameters
    ----------
    data : (m,) ndarray, of miil.eventcoinc_dtype
        Uncalibrated coincidence data
    system_shape : tuple or None
        The current shape of the Breast Panel system.  If None, then
        miil.default_system_shape is used.
    edep_type : str (Default = 'linear')
        'linear' or 'log' indicating the type of energy correction to do on
        the apds
    xtal_thres : int or tuple
        A threshold on the crystals.  If it is an int, all crystals with fewer
        counts than xtal_thres are excluded.  If it is a tuple, it applies a
        low and high threshold with xtal_thres[0] and xtal_thres[1],
        respectively.
    apd_thres : int or tuple
        Same as xtal_thres for apds.
    admm_iter: int (Default = 5)
        The number of iterations of ADMM to run on the fit to transform a
        least squares fit to a l1-norm fit.
    admm_rho: int (Default = None)
        Directly specify the rho parameter used by the ADMM algorithm.  If it
        is not specified, then it is estimated from admm_fwhm_exp and the
        estimated number of randoms. Always overrides admm_fwhm_exp and
        admm_invrho.
    admm_invrho: int (Default = None)
        Specify the rho parameter used by the ADMM algorithm by setting its
        inverse.  This is in the same units as 'dtf' which is nanoseconds.
    admm_fwhm_exp : float (Default = 16.0 ns)
        The expected timing resolution, in ns FWHM, of the calibrated
        distribution.  This is used to estimate the rho parameter fed into
        miil.opt.lad.
    xtal_global : bool (Default = False)
        data is assumed to be an event_coinc dtype, but the data can
        optionally have just crystal0, crystal1, and dtf fields.  If this is
        the case, then crystal0 and crystal1 are assumed to map to global
        crystal numbers.

    Returns
    -------
    tcal : (n,) ndarray, dtype=miil.tcal_dtype
        The timing calibration resulting from the algorithm.

    """
    if system_shape is None:
        system_shape = miil.default_system_shape
    if xtal_global:
        c0 = data['crystal0']
        c1 = data['crystal1']
        a0 = c0 // miil.no_crystals_per_apd(system_shape)
        a1 = c1 // miil.no_crystals_per_apd(system_shape)
    else:
        c0, c1 = miil.coinc_to_crystals(data, system_shape)
        a0, a1 = miil.coinc_to_apds(data, system_shape)

    c0_unique, c0_counts = np.unique(c0, return_counts=True)
    c1_unique, c1_counts = np.unique(c1, return_counts=True)
    a0_unique, a0_counts = np.unique(a0, return_counts=True)
    a1_unique, a1_counts = np.unique(a1, return_counts=True)

    c_vals = np.ones(miil.no_crystals(system_shape))
    a_vals = np.ones(miil.no_apds(system_shape))
    c_vals[miil.no_crystals_per_panel(system_shape):] *= -1
    a_vals[miil.no_apds_per_panel(system_shape):] *= -1

    if xtal_thres is not None:
        if isinstance(xtal_thres, int):
            c_vals[c0_unique[c0_counts < xtal_thres]] = 0
            c_vals[c1_unique[c1_counts < xtal_thres]] = 0
        elif isinstance(xtal_thres, tuple) and len(xtal_thres) == 2:
            c_vals[c0_unique[c0_counts < xtal_thres[0]]] = 0
            c_vals[c1_unique[c1_counts < xtal_thres[0]]] = 0
            c_vals[c0_unique[c0_counts > xtal_thres[1]]] = 0
            c_vals[c1_unique[c1_counts > xtal_thres[1]]] = 0
        else:
            raise ValueError('xtal_thres must be an int or tuple, len == 2')

    if apd_thres is not None:
        if isinstance(apd_thres, int):
            a_vals[c0_unique[a0_counts < apd_thres]] = 0
            a_vals[c1_unique[a1_counts < apd_thres]] = 0
        elif isinstance(apd_thres, tuple) and len(apd_thres) == 2:
            a_vals[c0_unique[a0_counts < apd_thres[0]]] = 0
            a_vals[c1_unique[a1_counts < apd_thres[0]]] = 0
            a_vals[c0_unique[a0_counts > apd_thres[1]]] = 0
            a_vals[c1_unique[a1_counts > apd_thres[1]]] = 0
        else:
            raise ValueError('apd_thres must be an int or tuple, len == 2')

    if edep_type == 'linear':
        A = csr_matrix((np.hstack((c_vals[c0], c_vals[c1],
            a_vals[a0], a_vals[a1],
            (data['E0'] - 511.0) * a_vals[a0],
            (data['E1'] - 511.0) * a_vals[a1])), (
                np.hstack((np.arange(data.size),) * 6),
                np.hstack((c0, c1,
                    a0 + miil.no_crystals(system_shape),
                    a1 + miil.no_crystals(system_shape),
                    a0 + 2 * miil.no_crystals(system_shape),
                    a1 + 2 * miil.no_crystals(system_shape)))
                )))
    elif edep_type == 'log':
        A = csr_matrix((
            np.hstack((c_vals[c0], c_vals[c1],
                       a_vals[a0], a_vals[a1],
                       (np.log(data['E0']) - np.log(511.0)) * a_vals[a0],
                       (np.log(data['E1']) - np.log(511.0)) * a_vals[a1])),
            (
                np.hstack((np.arange(data.size),) * 6),
                np.hstack((c0, c1,
                    a0 + miil.no_crystals(system_shape),
                    a1 + miil.no_crystals(system_shape),
                    a0 + 2 * miil.no_crystals(system_shape),
                    a1 + 2 * miil.no_crystals(system_shape)))
                )))
    else:
        raise ValueError('edep_type of {0} not supported. Only linear or log'.format(edep_type))

    if admm_rho is None:
        if admm_invrho is not None:
            admm_rho = 1.0 / admm_invrho
    else:
        admm_invrho = 1.0 / admm_rho

    if admm_rho is None:
        no_bins = 100
        n_raw, edge_raw = np.histogram(data['dtf'], bins=no_bins)
        rand_est = np.mean((n_raw[0], n_raw[-1])) * no_bins
        rand_frac_est = rand_est / np.sum(n_raw)
        print('Estimated randoms fraction: {0:0.1f}%'.format(rand_frac_est * 100))

        time_window = np.abs(edge_raw[0] - edge_raw[-1])
        sigma_exp = admm_fwhm_exp / 2.35
        admm_invrho = sigma_exp
        if rand_frac_est > 0:
            admm_invrho *= np.sqrt(-2 * np.log(sigma_exp * rand_frac_est * np.sqrt(2 * np.pi) /
                ((1 - rand_frac_est) * 2.0 * time_window)))
        else:
            admm_invrho = time_window
        admm_rho = 1.0 / admm_invrho

    print('Using rho: {0:0.3g}, corresponds to {1:0.1f}ns'.format(admm_rho, admm_invrho))
    print('Running {0} iterations of admm for l1-norm fit on {1} events'.format(admm_iter, c0.size))
    x = miil.opt.lad(A, data['dtf'],
            admm_rho, alpha=1.0, no_iter=admm_iter, abstol=1e-1, reltol=1e-2)
    tcal = np.empty(miil.no_crystals(system_shape), dtype=miil.tcal_dtype)
    no_crystals = miil.no_crystals(system_shape)
    tcal['offset'] = x[:no_crystals]
    tcal['offset'] += np.repeat(x[no_crystals:(2 * no_crystals)],
            miil.no_crystals_per_apd(system_shape))
    tcal['edep_offset'] = np.repeat(x[(2 * no_crystals):],
            miil.no_crystals_per_apd(system_shape))

    return tcal

def apply_tcal(data, tcal, system_shape=None, edep_type='linear',
        uv_period_ns=None, xtal_global=False):
    """
    Takes in a set of coincidence data and a timing calibration and returns a
    calibrated timestamp array.

    Parameters
    ----------
    data : (m,) ndarray, of miil.eventcoinc_dtype
        Uncalibrated coincidence data
    tcal : (n,) ndarray, dtype=miil.tcal_dtype, or str
        The timing calibration array, or filename from which it can be loaded
        using miil.load_time_calibration
    system_shape : tuple or None
        The current shape of the Breast Panel system.  If None, then
        miil.default_system_shape is used.
    edep_type : str (Default = 'linear')
        'linear' or 'log' indicating the type of energy correction to do on
        the apds
    uv_period_ns: float, str, or None
        The period of the UV sinusoid.  If str, we assume this is a BreastDAQ
        json file from which the UV period can be loaded using
        miil.load_uv_period.  If None, then the default of 1024.41 is used.
    xtal_global : bool (Default = False)
        data is assumed to be an event_coinc dtype, but the data can
        optionally have just crystal0, crystal1, and dtf fields.  If this is
        the case, then crystal0 and crystal1 are assumed to map to global
        crystal numbers.

    Returns
    -------
    dtf: (n,) ndarray, dtype=float
        The calibrated fine timestamp

    """
    if system_shape is None:
        system_shape = miil.default_system_shape

    # If tcal is a string, load in that time calibration first.
    if isinstance(tcal, str):
        tcal = load_time_calibration(tcal)
    if tcal.size != miil.no_crystals(system_shape):
        raise ValueError('Time calibration does not match system shape')

    if isinstance(uv_period_ns, str):
        uv_period_ns = miil.load_uv_period(uv_period_ns) * 1e9
    elif uv_period_ns is None:
        uv_period_ns = miil.default_uv_period_ns

    if xtal_global:
        c0 = data['crystal0']
        c1 = data['crystal1']
    else:
        c0, c1 = miil.coinc_to_crystals(data, system_shape)
    dtf = data['dtf'] - tcal['offset'][c0] + tcal['offset'][c1]

    if edep_type == 'linear':
        dtf -= (data['E0'] - 511.0) * tcal['edep_offset'][c0]
        dtf += (data['E1'] - 511.0) * tcal['edep_offset'][c1]
    elif edep_type == 'log':
        dtf -= (np.log(data['E0']) - np.log(511.0)) * tcal['edep_offset'][c0]
        dtf += (np.log(data['E1']) - np.log(511.0)) * tcal['edep_offset'][c1]
    else:
        raise ValueError('edep_type of "{0}" not supported. Only linear or log'.format(edep_type))

    # Protect our while loop by removing values that would mess up our
    # comparison.  We set them to the negative period, because this
    # bascially guarantees they will be time windowed out.  This strategy is
    # not practical for non coinc events.
    dtf[np.isnan(dtf)] = -uv_period_ns
    dtf[np.isinf(dtf)] = -uv_period_ns
    # wrap all of the events to the specified uv period.
    while np.any(dtf > uv_period_ns):
        dtf[dtf > uv_period_ns] -= uv_period_ns
    while np.any(dtf < -uv_period_ns):
        dtf[dtf < -uv_period_ns] += uv_period_ns
    return dtf
