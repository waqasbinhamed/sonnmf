import numpy as np


def proj_simplex(y):
    return np.maximum(y - np.max((np.cumsum(-1 * np.sort(-y, axis=0), axis=0) - 1) /
                              np.arange(1, y.shape[0] + 1).reshape(y.shape[0], 1), axis=0), 0)


def base(H, H_update_iters, M, W):
    for hit in range(H_update_iters):
        H = proj_simplex(H - ((W.T @ W) @ H - W.T @ M) / np.linalg.norm(W.T @ W, ord=2))
    return H


def precomputed_vars_and_nesterov_acc(H, H_update_iters, M, W, it):
    WtW = W.T @ W
    norm_WtW = np.linalg.norm(WtW, ord=2)
    Q = np.identity(W.shape[1]) - (WtW / norm_WtW)
    R = (W.T @ M) / norm_WtW
    V = H
    for hit in range(H_update_iters):
        H_old = H
        H = proj_simplex((Q @ V) + R)
        V = H + ((it - 1) / (it + 2)) * (H - H_old)
    return H
