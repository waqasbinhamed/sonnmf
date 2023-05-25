import numpy as np
from numba import jit
from sonnmf.numba.utils import ini_sonnmf, calculate_scores_and_report
from sonnmf.numba.update_H import base as base_H_func, precomputed_vars_and_nesterov_acc
from sonnmf.numba.update_W import base as base_W
EARLY_STOP_TOL = 1e-6


@jit(nopython=True)
def sonnmf(M, W, H, lam=0.0, gamma=0.0, itermin=100, itermax=10000, H_update_iters=1,
           W_update_iters=100, accelerate_H_update=False, early_stop=True, verbose=False):
    """
    Performs non-negative matrix factorization with penalties on W and H.

    Parameters:
    - M: the matrix to be factorized
    - W: the initial matrix for W
    - H: the initial matrix for H
    - lam: the penalty parameter for the group sparsity term
    - gamma: the penalty parameter for the non-negativity constraint on W
    - itermin: the minimum number of iterations to perform
    - itermax: the maximum number of iterations to perform
    - H_update_iters: the number of iterations to perform when updating H
    - wj_update_iters: the number of iterations to perform when updating the j-th column of W
    - early_stop: whether to use early stopping based on the change in objective function value
    - verbose: whether to print progress updates

    Returns:
    - W: the learned matrix W
    - H: the learned matrix H
    - fscores: an array of the objective function value at each iteration
    - gscores: an array of the group sparsity term value at each iteration
    - hscores: an array of the non-negativity constraint violation value at each iteration
    - total_scores: an array of the total objective function value at each iteration
    """

    fscores, gscores, hscores, total_scores = ini_sonnmf(itermax)
    scaled_lam_vals = np.empty((itermax,))
    scaled_gamma_vals = np.empty((itermax,))

    it = 0
    calculate_scores_and_report(H, M, W, fscores, gamma, gscores, hscores, it, lam, total_scores, verbose)
    # scaled_lam_vals[it] = (fscores[it] / gscores[it]) * lam if hscores[it] > 0 else lam
    # scaled_gamma_vals[it] = (fscores[it] / hscores[it]) * gamma if hscores[it] > 0 else gamma
    scaled_lam_vals[it] = lam
    scaled_gamma_vals[it] = gamma
    for it in range(1, itermax + 1):
        # update H
        if accelerate_H_update:
            H = precomputed_vars_and_nesterov_acc(H, H_update_iters, M, W)
        else:
            H = base_H_func(H, H_update_iters, M, W)

        # update W
        base_W(H, M, W, W_update_iters, scaled_gamma_vals[it-1], scaled_lam_vals[it-1])

        calculate_scores_and_report(H, M, W, fscores, scaled_gamma_vals[it-1], gscores, hscores, it,
                                    scaled_lam_vals[it-1], total_scores, verbose)
        # scaled_lam_vals[it] = min((fscores[it] / gscores[it]) * lam if gscores[it] > 0 else scaled_lam_vals[it-1], 1000000)
        # scaled_gamma_vals[it] = min((fscores[it] / hscores[it]) * gamma if hscores[it] > 0 else scaled_gamma_vals[it-1], 1000000)
        scaled_lam_vals[it] = lam
        scaled_gamma_vals[it] = gamma

        if early_stop and it > itermin:
            if abs(total_scores[it] - total_scores[it - 1]) / total_scores[it - 1] < EARLY_STOP_TOL:
                print(f'Early stopping condition reached at iteration {it}.')
                break
    scaled_lam_vals = np.concatenate((np.full(1, np.NaN), scaled_lam_vals[:it+1]))
    scaled_gamma_vals = np.concatenate((np.full(1, np.NaN), scaled_gamma_vals[:it+1]))
    return W, H, fscores[:it + 1], gscores[:it + 1], hscores[:it + 1], total_scores[:it + 1]
