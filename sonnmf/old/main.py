import numpy as np
from sonnmf.old.utils import sonnmf_ini, sonnmf_post_it
from sonnmf.old.update_H import base as update_H_base, precomputed_constants_with_nesterov as update_H_accelarated
from sonnmf.old.update_W import update_matrix as update_W_mat, update_matrix_with_penalty as update_W_with_penalty


def base(M, W, H, lam=0.0, w_update_method='proximal_averaging', h_update_accelarated=False, itermin=100, itermax=10000, h_update_iters=5, w_update_iters=1, early_stop=True, verbose=False, scale_reg=False):
    fscores, gscores, lambda_vals = sonnmf_ini(M, W, H, lam, itermax, scale_reg)

    for it in range(1, itermax + 1):
        # update H
        if h_update_accelarated:
            H = update_H_accelarated(M, W, H, iters=h_update_iters)
        else:
            H = update_H_base(M, W, H, iters=h_update_iters)

        # update W
        W = update_W_mat(M, W, H, lambda_vals[it - 1], method=w_update_method, iters=w_update_iters)

        fscores, gscores, lambda_vals, stop_now = sonnmf_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]


def with_nonneg_penalty(M, W, H, lam=0.0, gamma=0.0, h_update_accelarated=False, itermin=100, itermax=10000, h_update_iters=5, w_update_iters=1, early_stop=True, verbose=False, scale_reg=False):
    fscores, gscores, lambda_vals = sonnmf_ini(M, W, H, lam, itermax, scale_reg)

    for it in range(1, itermax + 1):
        # update H
        if h_update_accelarated:
            H = update_H_accelarated(M, W, H, iters=h_update_iters)
        else:
            H = update_H_base(M, W, H, iters=h_update_iters)

        # update W
        W = update_W_with_penalty(M, W, H, lam, gamma, iters=w_update_iters)

        fscores, gscores, lambda_vals, stop_now = sonnmf_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]
