import numpy as np
from numba import jit, prange


@jit(nopython=True)
def proj_simplex_1d(y):
    n = y.shape[0]
    sorted_y = np.sort(y)[::-1]
    cumsum_y = np.cumsum(sorted_y) - 1
    div_range = np.arange(1, n + 1)
    div_result = cumsum_y / div_range
    subtracted = y - max(div_result)
    result = np.maximum(subtracted, 0)
    return result


@jit(nopython=True)
def proj_simplex(y):
    result = np.empty_like(y)
    for i in range(y.shape[1]):
        result[:, i] = proj_simplex_1d(y[:, i])
    return result


@jit(nopython=True)
def base(H, H_update_iters, M, W):
    for it in range(H_update_iters):
        H = proj_simplex(H - ((W.T @ W) @ H - W.T @ M) / np.linalg.norm(W.T @ W, ord=2))
    return H


@jit(nopython=True)
def precomputed_vars_and_nesterov_acc(H, H_update_iters, M, W):
    WtW = W.T @ W
    norm_WtW = np.linalg.norm(WtW, ord=2)
    Q = np.identity(W.shape[1]) - (WtW / norm_WtW)
    R = (W.T @ M) / norm_WtW
    V = H
    for it in range(H_update_iters):
        H_old = H
        H = proj_simplex((Q @ V) + R)
        V = H + ((it - 1) / (it + 2)) * (H - H_old)
    return H