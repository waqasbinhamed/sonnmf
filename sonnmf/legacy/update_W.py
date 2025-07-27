import numpy as np

EPS = 1e-16
W_TOL = 1e-6

"""
This file contains multiple algorithms for updating the matrix W. These algorithms were tested during the development process to evaluate their performance and suitability for the final implementation.

Algorithms included:
1. Subgradient Method: A basic iterative approach using subgradients.

2. Nesterov Smoothing: Incorporates Nesterov acceleration for faster convergence.

3. ADMM (Alternating Direction Method of Multipliers): A method for solving optimization problems by breaking them into smaller subproblems.
   Reference: Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers."

4. Proximal Averaging: Uses proximal operators to enforce constraints and smoothness.

5. Proximal Averaging without Non-Negativity: A variant of proximal averaging that does not enforce non-negativity constraints. Useful for testing the impact of non-negativity constraints on optimization.

Each algorithm has its strengths and weaknesses, and the choice of algorithm depends on the specific requirements of the problem being solved.
"""


def calculate_function_value(c, diff_wis_norm, hj_norm_sq, MhT, lam):
    """
    Calculates the objective function value for the subgradient method.

    Parameters:
        c (np.ndarray): Current estimate of the solution.
        diff_wis_norm (np.ndarray): Norm differences for other components.
        hj_norm_sq (float): Squared norm of hj.
        MhT (np.ndarray): Residual matrix.
        lam (float): Regularization parameter.

    Returns:
        float: Objective function value.
    """
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + lam * np.sum(diff_wis_norm)


def perform_line_search(c, diff_wis, hj_norm_sq, MhT, lam, grad, beta=0.75, max_iters=1000):
    """
    Performs a simple line search to determine the step size.

    Parameters:
        c (np.ndarray): Current estimate of the solution.
        diff_wis (np.ndarray): Differences for other components.
        hj_norm_sq (float): Squared norm of hj.
        MhT (np.ndarray): Residual matrix.
        lam (float): Regularization parameter.
        grad (np.ndarray): Gradient of the objective function.
        beta (float): Step size reduction factor.
        max_iters (int): Maximum number of iterations.

    Returns:
        float: Step size.
    """
    t = 1
    k = 0
    val = calculate_function_value(c, diff_wis, hj_norm_sq, MhT, lam)
    while k < max_iters and calculate_function_value(c - t * grad, diff_wis, hj_norm_sq, MhT, lam) > val:
        t *= beta
        k += 1
    return t


def subgradient_method(W, Mj, z, hj, j, lam, max_iters=1000):
    """
    Subgradient method for updating W.

    Parameters:
        W (np.ndarray): Basis matrix.
        Mj (np.ndarray): Residual matrix.
        z (np.ndarray): Current estimate of the solution.
        hj (np.ndarray): Coefficient vector.
        j (int): Index of the column being updated.
        lam (float): Regularization parameter.
        max_iters (int): Maximum number of iterations.

    Returns:
        np.ndarray: Updated column of W.
    """
    def unit_norm_vector(size):
        tau = np.zeros((size, 1))
        tau[0] = 1
        return tau

    m = W.shape[0]
    new_z = z
    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T
    for k in range(max_iters):
        z = new_z

        diff_wis = np.delete(W, j, axis=1) - z
        diff_wis_norm = np.linalg.norm(diff_wis, axis=0)

        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[:, norm_mask] = diff_wis[:, norm_mask] / diff_wis_norm[norm_mask]
        tmp[:, ~norm_mask] = unit_norm_vector(m)
        grad = hj_norm_sq * z - MhT + np.sum(tmp, axis=1, keepdims=True)

        # Perform line search
        t = perform_line_search(z, diff_wis_norm, hj_norm_sq, MhT, lam, grad)
        new_z = np.maximum(z - t * grad, 0)

        if np.linalg.norm(z - new_z) / (np.linalg.norm(z) + EPS) < W_TOL:
            break
    return new_z


def nesterov_smoothing_method(W, Mj, new_z, hj, j, lam, mu=1, max_iters=1000):
    """
    Nesterov smoothing method for updating W.

    Parameters:
        W (np.ndarray): Basis matrix.
        Mj (np.ndarray): Residual matrix.
        new_z (np.ndarray): Current estimate of the solution.
        hj (np.ndarray): Coefficient vector.
        j (int): Index of the column being updated.
        lam (float): Regularization parameter.
        mu (float): Smoothing parameter.
        max_iters (int): Maximum number of iterations.

    Returns:
        np.ndarray: Updated column of W.
    """
    m = W.shape[0]

    wi_arr = np.delete(W, j, axis=1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T

    for k in range(max_iters):
        z = new_z

        tmp_arr = (z - wi_arr) / mu
        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        tmp_arr[:, norm_mask] = tmp_arr[:, norm_mask] / tmp_norm[norm_mask]

        grad = hj_norm_sq * z - MhT + np.sum(tmp_arr, axis=1).reshape(m, 1)
        t = perform_line_search(z, z - wi_arr, hj_norm_sq, MhT, lam, grad)

        new_z = np.maximum(z - t * grad, 0)

        if np.linalg.norm(z - new_z) / (np.linalg.norm(z) + EPS) < W_TOL:
            break
    return new_z


def admm_method(W, Mj, new_z, hj, j, lam, max_iters=1000):
    """
    ADMM method for updating W.

    Reference: Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers."

    Parameters:
        W (np.ndarray): Basis matrix.
        Mj (np.ndarray): Residual matrix.
        new_z (np.ndarray): Current estimate of the solution.
        hj (np.ndarray): Coefficient vector.
        j (int): Index of the column being updated.
        lam (float): Regularization parameter.
        max_iters (int): Maximum number of iterations.

    Returns:
        np.ndarray: Updated column of W.
    """
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    new_yi_arr = np.random.rand(m, r - 1)
    new_yf = np.random.rand(m, 1)
    new_y0 = np.random.rand(m, 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(max_iters):
        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = np.maximum(z - y0 / rho, 0)

        zeta_arr = z - yi_arr / rho
        tmp_arr = zeta_arr / lam - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - lam * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - lam * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / (np.linalg.norm(z) + EPS) < W_TOL:
            break

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def prox(mu, c, v):
    vc = v - c
    out = v - vc / np.max([1, np.linalg.norm(vc / mu)])
    return out


def prox_avg(W, Mj, hj, j, _lam):
    """
    Proximal averaging method for updating W.

    Parameters:
        W (np.ndarray): Basis matrix.
        Mj (np.ndarray): Residual matrix.
        hj (np.ndarray): Coefficient vector.
        j (int): Index of the column being updated.
        _lam (float): Regularization parameter.

    Returns:
        np.ndarray: Updated column of W.
    """
    _, rank = W.shape
    hj_norm_sq = hj @ hj.T

    w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
    prox_w_sum = 0
    for k in range(rank):
        if k != j:
            prox_w_sum += prox(_lam / ((hj_norm_sq * rank) + EPS), W[:, k: k + 1], w_bar)

    prox_in = np.minimum(w_bar, 0)
    return (prox_w_sum + prox_in) / rank


def prox_avg_without_nonneg(W, Mj, hj, j, lam):
    """
    Proximal averaging method for updating W without enforcing non-negativity constraints.

    This method computes the proximal operator for each component of W
    without applying non-negativity restrictions. It is useful for testing
    the impact of non-negativity constraints on the optimization process.

    Parameters:
        W (np.ndarray): Basis matrix.
        Mj (np.ndarray): Residual matrix.
        hj (np.ndarray): Coefficient vector.
        j (int): Index of the column being updated.
        lam (float): Regularization parameter.

    Returns:
        np.ndarray: Updated column of W.
    """
    _, rank = W.shape
    hj_norm_sq = hj @ hj.T

    # Compute the intermediate variable w_bar
    w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)

    # Compute the sum of proximal operators for other components
    prox_w_sum = 0
    for k in range(rank):
        if k != j:
            prox_w_sum += prox(lam / (hj_norm_sq * rank + EPS), W[:, k: k + 1], w_bar)

    # Return the average of the proximal operators
    return prox_w_sum / (rank - 1)


def update_w(method, H, M, W, num_iters, lam):
    """
    Updates the matrix W using the specified method.

    Parameters:
        method (str): The method to use for updating W ('subgradient', 'nesterov_smoothing', etc.).
        H (np.ndarray): Coefficient matrix.
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix to be updated.
        num_iters (int): Number of iterations for updating W.
        lam (float): Regularization parameter.

    Returns:
        np.ndarray: Updated matrix W.
    """
    _, rank = W.shape

    # TODO: add convergence check
    for it in range(num_iters):
        for j in range(rank):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]
            Mj = M - W @ H + wj @ hj
            if method == 'proximal_avg_with_indicator_func':
                W[:, j: j + 1] = wj = prox_avg(W, Mj, hj, j, lam)
            elif method == 'admm':
                W[:, j: j + 1] = wj = admm_method(W, Mj, wj, hj, j, lam)
            elif method == 'subgradient':
                W[:, j: j + 1] = wj = subgradient_method(W, Mj, wj, hj, j, lam)
            elif method == 'nesterov_smoothing':
                W[:, j: j + 1] = wj = nesterov_smoothing_method(W, Mj, wj, hj, j, lam)
            else:
                raise ValueError('Unknown method specified for solving the w_j subproblem.')
    return W
