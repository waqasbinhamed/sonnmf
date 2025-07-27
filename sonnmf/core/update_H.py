import numpy as np

def project_to_simplex(y):
    """
    Projects a vector onto the probability simplex.
    This ensures that the elements of the vector are non-negative and sum to 1.

    Parameters:
        y (np.ndarray): Input vector to be projected.

    Returns:
        np.ndarray: Projected vector.
    """
    return np.maximum(y - np.max((np.cumsum(-1 * np.sort(-y, axis=0), axis=0) - 1) /
                              np.arange(1, y.shape[0] + 1).reshape(y.shape[0], 1), axis=0), 0)

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
        H = project_to_simplex(H - ((W.T @ W) @ H - W.T @ M) / np.linalg.norm(W.T @ W, ord=2))
    return H

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
        H = project_to_simplex((Q @ V) + R)
        # Apply Nesterov acceleration
        V = H + ((t - 1) / (t + 2)) * (H - H_old)
    return H
