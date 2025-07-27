import numpy as np

def calculate_fscore(M, W, H):
    """
    Calculates the Frobenius norm of the difference between M and WH.

    Reference: Equation (1) in [arXiv:2407.00706]

    Parameters:
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix.
        H (np.ndarray): Coefficient matrix.

    Returns:
        float: Frobenius norm of the difference.
    """
    return 0.5 * np.linalg.norm(M - np.dot(W, H), 'fro') ** 2

def calculate_gscore(W):
    """
    Calculates the group sparsity term for W.

    Reference: Equation (2) in [arXiv:2407.00706]

    Parameters:
        W (np.ndarray): Basis matrix.

    Returns:
        float: Group sparsity score.
    """
    rank = W.shape[1]
    gscore = 0
    for i in range(rank - 1):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def calculate_scores_and_report(H, M, W, fscores, gscores, it, lam, total_scores, verbose):
    """
    Calculates and reports the scores for the current iteration.

    Parameters:
        H (np.ndarray): Coefficient matrix.
        M (np.ndarray): Input data matrix.
        W (np.ndarray): Basis matrix.
        fscores (np.ndarray): Array to store Frobenius norm scores.
        gscores (np.ndarray): Array to store group sparsity scores.
        it (int): Current iteration index.
        lam (float): Regularization parameter for group sparsity.
        total_scores (np.ndarray): Array to store total objective scores.
        verbose (bool): Whether to print progress updates.
    """
    fscores[it] = calculate_fscore(M, W, H)
    gscores[it] = calculate_gscore(W)
    total_scores[it] = fscores[it] + lam * gscores[it]
    if verbose:
        print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]}, total={total_scores[it]}')


def ini_sonnmf(itermax):
    """
    Initializes score arrays for SONNMF.

    Parameters:
        itermax (int): Maximum number of iterations.

    Returns:
        tuple: Initialized arrays for fscores, gscores, and total_scores.
    """
    fscores = np.empty((itermax + 1,))
    gscores = np.empty((itermax + 1,))
    total_scores = np.empty((itermax + 1,))
    return fscores, gscores, total_scores
