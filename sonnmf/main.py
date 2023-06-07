import numpy as np
from sonnmf.utils import ini_sonnmf, calculate_scores_and_report
from sonnmf.update_H import base as base_H_func, precomputed_vars_and_nesterov_acc
from sonnmf.update_W import base as base_W

EARLY_STOP_TOL = 1e-6


def update_O(M, W, H, O, beta):
    M_bar = M - W @ H
    for j in range(O.shape[1]):
        mj = M_bar[:, j:j + 1]
        O[:, j:j + 1] = mj / max(1, np.linalg.norm(mj / beta))
    return O


def sonnmf(M, W, H, O, lam=0.0, gamma=0.0, beta=0.0, itermin=500, itermax=10000, H_update_iters=1,
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

    fscores, gscores, hscores, iscores, total_scores = ini_sonnmf(itermax)

    it = 0
    calculate_scores_and_report(H, M, W, O, fscores, gamma, gscores, hscores, iscores, it, lam, beta, total_scores, verbose)
    for it in range(1, itermax + 1):

        M_o = M - O

        # update H
        if accelerate_H_update:
            H = precomputed_vars_and_nesterov_acc(H, H_update_iters, M_o, W)
        else:
            H = base_H_func(H, H_update_iters, M_o, W)

        # update W
        base_W(H, M_o, W, W_update_iters, gamma, lam)

        # update O
        O = update_O(M, W, H, O, beta)

        if it % 50 == 0:
            calculate_scores_and_report(H, M, W, O, fscores, gamma, gscores, hscores, iscores, it, lam, beta, total_scores, verbose)
        if early_stop and it > itermin:
            if abs(total_scores[it] - total_scores[it - 1]) / total_scores[it - 1] < EARLY_STOP_TOL:
                print(f'Early stopping condition reached at iteration {it}.')
                break

    return W, H, fscores[:it + 1], gscores[:it + 1], hscores[:it + 1], iscores[:it + 1], total_scores[:it + 1]
