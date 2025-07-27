import numpy as np

EPS = 1e-16

def proximal_operator(mu, c, v):
    """
    Calculates the proximal operator of v with respect to mu and c.

    Parameters:
        mu (float): Regularization parameter.
        c (np.ndarray): Offset vector.
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Result of the proximal operator.
    """
    vc = v - c
    return v - vc / max(1, np.linalg.norm(vc / mu))

def update_w_basic(H, M, W, num_iters, gamma, lam):
    """
    Updates the matrix W using a basic iterative approach.

    Reference: Algorithm 3 in [arXiv:2407.00706]

    Parameters:
        H (np.ndarray): Coefficient matrix.
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix to be updated.
        num_iters (int): Number of iterations for updating W.
        gamma (float): Regularization parameter for sparsity.
        lam (float): Regularization parameter for smoothness.

    Returns:
        np.ndarray: Updated matrix W.
    """
    m, rank = W.shape
    for it in range(num_iters):
        for j in range(rank):
            hj = H[j:j + 1, :]
            hj_norm_sq = hj @ hj.T
            wj = W[:, j:j + 1]

            # Compute the residual matrix Mj
            Mj = M - W @ H + wj @ hj

            # Compute the intermediate variables
            w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
            alpha = (rank - 1) * lam + gamma

            # Compute the proximal penalty term
            prox_pen = np.median(
                np.hstack((w_bar + (gamma / (hj_norm_sq + EPS)), np.zeros((m, 1)), w_bar)),
                axis=1,
                keepdims=True
            )

            # Compute the sum of proximal operators for other components
            prox_w_sum = np.sum(
                np.array([
                    proximal_operator(lam / (hj_norm_sq + EPS), W[:, k:k + 1], w_bar)
                    for k in range(rank) if k != j
                ]),
                axis=0
            )

            # Update the j-th column of W
            W[:, j:j + 1] = (lam * prox_w_sum + gamma * prox_pen) / alpha
    return W
