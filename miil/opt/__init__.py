import numpy as np
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg.interface import aslinearoperator

def mlem(y, A, no_iter, verbose=False,
         ret_iter_x=0, ret_iter_y=0, ret_norm_r=False, ret_objective=False,
         AT_ones=None, x0=None, inverse_thres=0.0):
    '''
    Maximizes the log-likelihood for a Poisson Random Varible.  y is the
    observed poisson random variable, modeled by A * x.  Maximizes
        f(x) = 1^T Ax - y^T ln(Ax)
    Taken from Jingyu Cui's Thesis, "Fast and Accurate PET Image Reconstruction
    on Parallel Architectures," 2013.

    Parameters
    ----------
    y : (m,) array-like
        Observed Poisson variable
    A : (m,n) matrix, sparse matrix, or LinearOperator
        System Model
    no_iter : int scalar
        Number of update iterations to perform.
    verbose : boolean (Default = False), optional
        If the relative residual norm and objective should be printed out
        each iteration.
    ret_iter_x : int scalar, optional
        Return the inter-iteration history of x every from ret_iter_x
        iterations.  If zero, the inter-iteration history of x is not returned
    ret_iter_y : int scalar, optional
        Return the inter-iteration history of y_bar (the model) every from
        ret_iter_y iterations.  If zero, the inter-iteration history of y_bar
        is not returned.
    ret_norm_r : boolean (Default = False), optional
        Return the norm of the relative residual from iteration.
    ret_objective : boolean (Default = False), optional
        Return the objective function value from iteration.
    AT_ones : (n,) array-like, optional
        The result of A^T 1 can be provided to avoid the computation.  AT_ones
        is used to normalize the error backpropogation step.
    x0 : (n,) array-like, optional
        Override the default model initialization.  Default is the number of
        counts in y, divided by n for each x0.
    inverse_thres : float scalar
        Zeros out small inverse values for the error propogation.  The inverse
        of errors are backprojected with A^T.  Small model values can cause the
        value to explode unnecessarily.

    Returns
    -------
    x : (n,) ndarray
        The weights resulting from the algorithm.
    x_history : (variable, n) ndarray, optional
        The x from each iteration specified by ret_iter_x, plus the final value.
    y_history : (variable, m) ndarray, optional
        The y_bar from each iteration specified by ret_iter_y, plus the final
        model value.
    norm_r_history : (no_iter,) ndarray, optional
        The norm of the relative residual from iteration
    objective_history : (no_iter,) ndarray, optional
        The objective function value from iteration

    '''
    A = aslinearoperator(A)
    y = np.asarray(y, dtype=A.dtype).squeeze()
    if AT_ones is None:
        AT_ones = A.rmatvec(np.ones(A.shape[0]))
    else:
        AT_ones = np.asarray(AT_ones, dtype=A.dtype).squeeze()
        if AT_ones.shape != (A.shape[1],):
            raise ValueError('AT_ones is not shaped (%d,)' % A.shape[1])

    if x0 is None:
        # Initialize it to uniform weights where the total counts would match
        x = np.ones(A.shape[1], dtype=A.dtype) * (y.sum() / A.shape[1])
    else:
        x = np.asarray(x0, dtype=A.dtype).squeeze()

    norm_y = np.linalg.norm(y)

    # Save every history_idx iterations, and the last one
    save_model_idx = np.zeros(no_iter + 1, dtype=bool)
    if ret_iter_y > 0:
        save_model_idx[(np.arange(no_iter + 1) % ret_iter_y) == 0] = True
        save_model_idx[-1] = True

    save_x_idx = np.zeros(no_iter + 1, dtype=bool)
    if ret_iter_x > 0:
        save_x_idx[(np.arange(no_iter + 1) % ret_iter_x) == 0] = True
        save_x_idx[-1] = True

    history_size_model = np.sum(save_model_idx)
    history_size_x = np.sum(save_x_idx)

    x_history = np.zeros((history_size_x, A.shape[1]))
    model_history = np.zeros((history_size_model, A.shape[0]))

    # These vectors are really small in comparison, so save them every
    # iteration, and worry about returning them later.
    norm_r_history = np.zeros((no_iter + 1,))
    objective_history = np.zeros((no_iter + 1,))
    history_count_x = 0
    history_count_model = 0

    for iter_no in xrange(no_iter + 1):
        if verbose:
            print '%02d: ' % (iter_no),
        model = A.matvec(x)

        norm_r = np.linalg.norm(model - y) / norm_y
        if verbose:
            print 'rel_norm = ', norm_r,

        objective = model[model > 0].astype(np.float128).sum() - \
                    (y[model > 0] * np.log(model[model > 0])
                    ).astype(np.float128).sum()
        if verbose:
            print '  objective = ', objective

        if save_x_idx[iter_no]:
            x_history[history_count_x, :] = x.copy()
            history_count_x += 1
        if save_model_idx[iter_no]:
            model_history[history_count_model, :] = model.copy()
            history_count_model += 1
        norm_r_history[iter_no] = norm_r
        objective_history[iter_no] = objective

        # We loop for no_iter + 1 so that we can calculate the model error
        # for the final iteration.
        if iter_no == no_iter:
            continue

        error = np.zeros(A.shape[0], dtype=A.dtype)
        error[model > 0] = y[model > 0] / model[model > 0]
        error[model <= 0] = 0
        if inverse_thres > 0:
            error[error > 1.0 / inverse_thres] = 0

        error_bp = A.rmatvec(error)

        update = np.zeros(A.shape[1], dtype=A.dtype)
        update[AT_ones > 0] = error_bp[AT_ones > 0] / AT_ones[AT_ones > 0]
        update[AT_ones <= 0] = 0

        update[update < 0] = 0

        x *= update

    ret = [x,]
    if ret_iter_x > 0:
        ret.append(x_history)
    if ret_iter_y > 0:
        ret.append(model_history)
    if ret_norm_r:
        ret.append(norm_r_history)
    if ret_objective:
        ret.append(objective_history)

    if len(ret) == 1:
        ret = ret[0]

    return ret

def _shrinkage(a, kappa):
    """
    The shrinkage operation as described by Boyd, et al. 2011 for the L1 norm.
    It returns:
        pos(a - kappa) - pos(-a - kappa)
    Where pos returns a 0 for negative numbers.

    Parameters
    ----------
    a : array_like
        values to be shrunk pairwise
    kappa : scalar
        The shrinkage scale

    Returns
    -------
    val : ndarray
        The result
    """
    a = np.asarray(a)
    tmp = a - kappa
    val = np.max(np.stack((tmp, np.zeros(a.shape))), axis=0)
    tmp = - a - kappa
    val -= np.max(np.stack((tmp, np.zeros(a.shape))), axis=0)
    return val

def lad(A, b, rho,
        alpha=1.3, verbose=True, ret_hist=False,
        abstol=1e-1, reltol=1e-2, no_iter=30,
        lsmr_atol=1e-6, lsmr_btol=1e-6, lsmr_conlim=1e8, lsmr_iter=1000):
    """
    A port of the demonstration by Boyd, et al. in Matlab which was modified to
    to use LSMR for the x-step.  Solves the following problem via ADMM:

       ``minimize     ||Ax - b||_1``

    More information can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    rho : scalar
        The augmented Lagrangian parameter, which will need to be set based
        upon the application.  Typical values have been around 0.05.
    alpha : scalar
        The over-relaxation parameter, which should be above 1.0 and are
        typically below 1.8.
    verbose : bool
        Print out the progress per iteration
    ret_hist : bool
        Return the history of ADMM and LSQR per iteration.
    abstol : scalar
        Absolute tolerance for the ADMM algorithm.  Not used currently.
    reltol : scalar
        Relative tolerance for the ADMM algorithm.  Not used currently.
    no_iter : int
        Maximum number of iterations of ADMM to be run.
    lsmr_atol : scalar
        The relative tolerance for ``A`` in the LSMR algorithm.
    lsmr_btol : scalar
        The relative tolerance for ``b`` in the LSMR algorithm.
    lsmr_conlim : scalar
        The condition limit for ``A`` in the LSMR algorithm.
    lsmr_iter : int
        Maximum number of iterations of LSMR to be run each ADMM iteration.

    Parameters
    ----------
    x : array_like, shape (n,)
        The L1 norm estimate of ``x`` in ``Ax = b``

    """
    A = aslinearoperator(A)
    b = np.asarray(b)

    m = A.shape[0]
    n = A.shape[1]

    x = np.zeros(n);
    z = np.zeros(m);
    u = np.zeros(m);
    Ax = np.zeros(m)

    if ret_hist:
        history = dict()
        history['x'] = np.zeros((no_iter, n))
        history['r_norm'] = np.zeros(no_iter)
        history['s_norm'] = np.zeros(no_iter)
        history['eps_prim'] = np.zeros(no_iter)
        history['eps_dual'] = np.zeros(no_iter)
        history['objective'] = np.zeros(no_iter)
        history['lsmr'] = dict()
        history['lsmr']['flag'] = np.zeros(no_iter)
        history['lsmr']['iter'] = np.zeros(no_iter)
        history['lsmr']['normr'] = np.zeros(no_iter)
        history['lsmr']['normar'] = np.zeros(no_iter)
        history['lsmr']['norma'] = np.zeros(no_iter)
        history['lsmr']['conda'] = np.zeros(no_iter)
        history['lsmr']['normx'] = np.zeros(no_iter)

    if verbose:
        print '%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % (
            'iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective')

    for iter_no in range(no_iter):
        q = b + z - u
        # Perform LSMR on the residual and add it to x as suggested by scipy
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.linalg.lsqr.html
        # to perform a warm-start on the algorithm.
        # TODO: Warm start was added in v1.0, so change this to use that
        # feature
        r0 = q - Ax
        (dx, lsmr_flag, lsmr_iter, lsmr_normr, lsmr_normar, lsmr_norma,
             lsmr_conda, lsmr_normx) = \
             lsmr(A, r0, 0, lsmr_atol, lsmr_btol, lsmr_conlim, lsmr_iter)
        x += dx

        if ret_hist:
            history['x'][iter_no, :] = x.copy()
            history['lsmr']['flag'][iter_no] = lsmr_flag
            history['lsmr']['iter'][iter_no] = lsmr_iter
            history['lsmr']['normr'][iter_no] = lsmr_normr
            history['lsmr']['normar'][iter_no] = lsmr_normar
            history['lsmr']['norma'][iter_no] = lsmr_norma
            history['lsmr']['conda'][iter_no] = lsmr_conda
            history['lsmr']['normx'][iter_no] = lsmr_normx

        zold = z
        Ax = A.matvec(x)
        Ax_hat = alpha * Ax + (1 - alpha) * (zold + b)
        z = _shrinkage(Ax_hat - b + u, 1 / rho)

        u = u + (Ax_hat - z - b)

        # Calculate the iteration progress
        objval = np.linalg.norm(z, 1)
        r_norm = np.linalg.norm(Ax - z - b)
        s_norm = np.linalg.norm(rho * A.rmatvec(zold - z))

        # And the stopping criterion
        eps_prim = abstol * np.sqrt(n) + \
                   reltol * np.max((np.linalg.norm(Ax),
                                     np.linalg.norm(-z),
                                     np.linalg.norm(b)))
        eps_dual = abstol * np.sqrt(n) + \
                   reltol * np.linalg.norm(rho * A.rmatvec(u))

        if ret_hist:
            history['r_norm'][iter_no] = r_norm
            history['s_norm'][iter_no] = s_norm
            history['eps_prim'][iter_no] = eps_prim
            history['eps_dual'][iter_no] = eps_dual
            history['objective'][iter_no] = objval

        if (r_norm < eps_prim) & (s_norm < eps_dual):
            break

        if verbose:
            print '%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % (
                iter_no, r_norm, eps_prim, s_norm, eps_dual, objval)

    if ret_hist:
        return x, history
    else:
        return x

