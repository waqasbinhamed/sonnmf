import numpy as np
from numba import jit, prange


EPS = 1e-16


@jit(nopython=True, parallel=True)
def rowwise_median(matrix):
    """Compute the median value for each row in the matrix."""
    num_rows = matrix.shape[0]
    medians = np.zeros(num_rows)

    for i in prange(num_rows):
        medians[i] = np.median(matrix[i])

    return medians.reshape(-1, 1)


@jit(nopython=True, parallel=True)
def prox(mu, c, v):
    vc = v - c
    return v - vc / max(1, np.linalg.norm(vc / mu))


@jit(nopython=True, parallel=True)
def base(H, M, W, W_update_iters, gamma, lam):
    m, rank = W.shape
    for it in range(W_update_iters):
        Mj = M - W @ H
        for j in range(rank):
            hj = H[j:j + 1, :]

            hj_norm_sq = np.linalg.norm(hj) ** 2
            wj = W[:, j:j + 1]

            Mj = Mj + wj @ hj
            w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
            alpha = (rank - 1) * lam + gamma
            prox_pen = rowwise_median(np.hstack((w_bar + (gamma / (hj_norm_sq + EPS)), np.zeros((m, 1)), w_bar)))

            prox_w_sum = np.zeros_like(wj)
            for k in range(rank):
                if k != j:
                    prox_w_sum += prox(lam / (hj_norm_sq + EPS), W[:, k:k + 1], w_bar)

            W[:, j:j + 1] = (lam * prox_w_sum + gamma * prox_pen) / alpha
            Mj = Mj - wj @ hj
    return W
