import numpy as np
from scipy.sparse.linalg.interface import aslinearoperator

def mlem(y, A, no_iter, verbose=False,
         ret_iter_x=0, ret_iter_y=0, ret_norm_r=False, ret_objective=False,
         AT_ones=None, x0=None):
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
    y = np.atleast_1d(np.asarray(y, dtype=A.dtype))
    if AT_ones is None:
        AT_ones = A.rmatvec(np.ones(A.shape[0]))
    else:
        AT_ones = np.atleast_1d(np.asarray(AT_ones, dtype=A.dtype))

    if x0 is None:
        # Initialize it to uniform weights where the total counts would match
        x = np.ones(A.shape[1], dtype=A.dtype) * (y.sum() / A.shape[1])
    else:
        x = np.atleast_1d(np.asarray(x0, dtype=A.dtype))

    error = np.zeros(A.shape[0], dtype=A.dtype)
    update = np.zeros(A.shape[1], dtype=A.dtype)
    norm_y =  np.linalg.norm(y)

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

        objective = model[model > 0].sum() - \
                    (y[model > 0] * np.log(model[model > 0])).sum()
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

        error[model > 0] = y[model > 0] / model[model > 0]
        error[model <= 0] = 0

        error_bp = A.rmatvec(error)

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
