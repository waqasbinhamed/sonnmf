import numpy as np
from sonnmf.utils import non_neg

EPS = 1e-12
W_TOL = 1e-6


def calculate_func_val(c, diff_wis_norm, hj_norm_sq, MhT, lam):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + lam * np.sum(diff_wis_norm)


def line_search(c, diff_wis, hj_norm_sq, MhT, lam, grad, beta=0.75, itermax=1000):
    t = 1
    k = 0
    val = calculate_func_val(c, diff_wis, hj_norm_sq, MhT, lam)
    while k < itermax and calculate_func_val(c - t * grad, diff_wis, hj_norm_sq, MhT, lam) > val:
        t *= beta
        k += 1
    return t


def subgrad(W, Mj, z, hj, j, lam, itermax=1000):
    def unit_norm_vec(sz):
        tau = np.zeros((sz, 1))
        tau[0] = 1
        return tau

    m = W.shape[0]
    new_z = z
    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T
    for k in range(itermax):
        z = new_z

        diff_wis = np.delete(W, j, axis=1) - z
        diff_wis_norm = np.linalg.norm(diff_wis, axis=0)

        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[:, norm_mask] = diff_wis[:, norm_mask] / diff_wis_norm[norm_mask]
        tmp[:, ~norm_mask] = unit_norm_vec(m)
        grad = hj_norm_sq * z - MhT + np.sum(tmp, axis=1, keepdims=True)

        # simple line search
        t = line_search(z, diff_wis_norm, hj_norm_sq, MhT, lam, grad)
        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / (np.linalg.norm(z) + EPS) < W_TOL:
            break
    return new_z


def nesterov_smoothing(W, Mj, new_z, hj, j, lam, mu=1, itermax=1000):
    m = W.shape[0]

    wi_arr = np.delete(W, j, axis=1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T

    for k in range(itermax):
        z = new_z

        tmp_arr = (z - wi_arr) / mu
        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        tmp_arr[:, norm_mask] = tmp_arr[:, norm_mask] / tmp_norm[norm_mask]

        grad = hj_norm_sq * z - MhT + np.sum(tmp_arr, axis=1).reshape(m, 1)
        t = line_search(z, z - wi_arr, hj_norm_sq, MhT, lam, grad)

        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / (np.linalg.norm(z) + EPS) < W_TOL:
            break
    return new_z


def admm(W, Mj, new_z, hj, j, _lam, itermax=1000):
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    new_yi_arr = np.random.rand(m, r - 1)
    new_yf = np.random.rand(m, 1)
    new_y0 = np.random.rand(m, 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = non_neg(z - y0 / rho)

        zeta_arr = z - yi_arr / rho
        tmp_arr = zeta_arr / _lam - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - _lam * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - _lam * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < W_TOL:
            break

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def prox(a, c, v):
    vc = v - c
    out = v - vc / np.max([1, np.linalg.norm(vc / a)])
    return out


def prox_avg(W, Mj, hj, j, _lam):
    _, rank = W.shape
    hj_norm_sq = hj @ hj.T

    w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
    prox_w_sum = 0
    for k in range(rank):
        if k != j:
            prox_w_sum += prox(_lam / (hj_norm_sq * rank + EPS), W[:, k: k + 1], w_bar)

    prox_in = non_neg(w_bar)
    return (prox_w_sum + prox_in) / rank


def update_matrix   (M, W, H, _lam, method='proximal_averaging', iters=1):
    _, rank = W.shape

    # TODO: add convergence check
    for it in range(iters):
        Mj = M - W @ H
        for j in range(rank):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]
            Mj = Mj + wj @ hj
            if method == 'proximal_averaging':
                W[:, j: j + 1] = wj = prox_avg(W, Mj, hj, j, _lam)
            elif method == 'admm':
                W[:, j: j + 1] = wj = admm(W, Mj, wj, hj, j, _lam)
            elif method == 'subgradient':
                W[:, j: j + 1] = wj = subgrad(W, Mj, wj, hj, j, _lam)
            elif  method == 'nesterov_smoothing':
                W[:, j: j + 1] = wj = nesterov_smoothing(W, Mj, wj, hj, j, _lam)
            else:
                raise ValueError('Unknown method specified for solving the w_j subproblem. Please specify one of the following: proximal_averaging, admm, subgradient, nesterov_smoothing')

            Mj = Mj - wj @ hj
    
    return W