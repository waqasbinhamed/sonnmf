import numpy as np

EPS = 1e-16


def prox(mu, c, v):
    """Calculates the proximal operator of v with respect to mu and c."""
    vc = v - c
    return v - vc / max(1, np.linalg.norm(vc / mu))


def base(H, M, W, W_update_iters, gamma, lam, m, rank):
    for wit in range(W_update_iters):
        for j in range(rank):
            hj = H[j:j + 1, :]
            hj_norm_sq = hj @ hj.T
            wj = W[:, j:j + 1]

            Mj = M - W @ H + wj @ hj
            w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
            alpha = (rank - 1) * lam + gamma
            prox_pen = np.median(np.hstack((w_bar + (gamma / (hj_norm_sq + EPS)), np.zeros((m, 1)), w_bar)), axis=1,
                                 keepdims=True)
            prox_w_sum = np.sum(
                np.array([prox(lam / (hj_norm_sq + EPS), W[:, k:k + 1], w_bar) for k in range(rank) if k != j]), axis=0)

            W[:, j:j + 1] = (lam * prox_w_sum + gamma * prox_pen) / alpha