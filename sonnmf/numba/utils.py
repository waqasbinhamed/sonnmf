import numpy as np
from numba import jit, prange


@jit(nopython=True)
def frobenius_norm(A):
    """Calculates the Frobenius norm of a matrix A."""
    m, n = A.shape
    norm = 0.0

    for i in range(m):
        for j in range(n):
            norm += A[i, j] * A[i, j]

    return np.sqrt(norm)


@jit(nopython=True)
def calculate_fscore(M, W, H):
    """Calculates the Frobenius norm of the difference between M and WH."""
    diff = M - np.dot(W, H)
    return 0.5 * frobenius_norm(diff) ** 2


@jit(nopython=True)
def calculate_gscore(W):
    """Calculates the sum of the norm of the columns of W."""
    rank = W.shape[1]
    gscore = 0.0

    for i in range(rank - 1):
        norm_sum = 0.0
        for j in range(i + 1, rank):
            diff = W[:, i] - W[:, j]
            norm_sum += np.sqrt(np.dot(diff, diff))
        gscore += norm_sum

    return gscore


@jit(nopython=True)
def calculate_hscore(W):
    """Calculates the negative sum of the minimum of each element of W and 0."""
    return -np.sum(np.minimum(W, 0))


@jit(nopython=True)
def calculate_scores_and_report(H, M, W, fscores, gamma, gscores, hscores, it, lam, total_scores, verbose):
    fscores[it] = calculate_fscore(M, W, H)
    gscores[it] = calculate_gscore(W)
    hscores[it] = calculate_hscore(W)
    total_scores[it] = fscores[it] + lam * gscores[it] + gamma * hscores[it]
    if verbose:
        print(it, fscores[it], gscores[it], hscores[it], total_scores[it])
    # if verbose:
    #     print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]}, h={hscores[it]}, total={total_scores[it]}')


@jit(nopython=True)
def ini_sonnmf(itermax):
    fscores = np.empty((itermax + 1,))
    gscores = np.empty((itermax + 1,))
    hscores = np.empty((itermax + 1,))
    total_scores = np.empty((itermax + 1,))
    return fscores, gscores, hscores, total_scores
