import numpy as np
from numba import jit


@jit(nopython=True)
def project_to_simplex(y):
    """
    Projects a vector onto the probability simplex.
    This ensures that the elements of the vector are non-negative and sum to 1.

    Parameters:
        y (np.ndarray): Input vector to be projected.

    Returns:
        np.ndarray: Projected vector.
    """
    n = y.shape[0]
    sorted_y = np.sort(y)[::-1]
    cumsum_y = np.cumsum(sorted_y) - 1
    div_range = np.arange(1, n + 1)
    div_result = cumsum_y / div_range
    subtracted = y - max(div_result)
    return np.maximum(subtracted, 0)


@jit(nopython=True)
def project_columns_to_simplex(y):
    """
    Projects each column of a matrix onto the simplex.

    Parameters:
        y (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Matrix with each column projected onto the simplex.
    """
    result = np.empty_like(y)
    for i in range(y.shape[1]):
        result[:, i] = project_to_simplex(y[:, i])
    return result


@jit(nopython=True)
def update_h_basic(H, num_iters, M, W):
    """
    Updates the matrix H using a basic iterative approach.

    Reference: Algorithm 1 in [arXiv:2407.00706]

    Parameters:
        H (np.ndarray): Initial matrix H.
        num_iters (int): Number of iterations for updating H.
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix.

    Returns:
        np.ndarray: Updated matrix H.
    """
    for t in range(num_iters):
        # Compute the gradient and project onto the simplex
        H = project_columns_to_simplex(H - ((W.T @ W) @ H - W.T @ M) / np.linalg.norm(W.T @ W, ord=2))
    return H


@jit(nopython=True)
def update_h_nesterov(H, num_iters, M, W):
    """
    Updates the matrix H using precomputed variables and Nesterov acceleration.

    Reference: Algorithm 2 in [arXiv:2407.00706]

    Parameters:
        H (np.ndarray): Initial matrix H.
        num_iters (int): Number of iterations for updating H.
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix.

    Returns:
        np.ndarray: Updated matrix H.
    """
    # Precompute W^T * W and its norm
    WtW = W.T @ W
    norm_WtW = np.linalg.norm(WtW, ord=2)

    # Precompute Q and R for the update
    Q = np.identity(W.shape[1]) - (WtW / norm_WtW)
    R = (W.T @ M) / norm_WtW

    # Initialize variables for Nesterov acceleration
    V = H
    for t in range(num_iters):
        H_old = H
        # Update H using the precomputed variables
        H = project_columns_to_simplex((Q @ V) + R)
        # Apply Nesterov acceleration
        V = H + ((t - 1) / (t + 2)) * (H - H_old)
    return H
