import numpy as np


def calculate_fscore(M, W, H, O):
    """Calculates the Frobenius norm of the difference between M and WH."""
    return 0.5 * np.linalg.norm(M - np.dot(W, H) - O, 'fro') ** 2


def calculate_gscore(W):
    """Calculates the sum of the norm of the columns of W."""
    rank = W.shape[1]
    gscore = 0
    for i in range(rank - 1):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def calculate_hscore(W):
    """Calculates the negative sum of the minimum of each element of W and 0."""
    return -np.sum(np.minimum(W, 0))


def calculate_scores_and_report(H, M, W, O, fscores, gamma, gscores, hscores, iscores, it, lam, beta, total_scores, verbose):
    fscores[it] = calculate_fscore(M, W, H, O)
    gscores[it] = calculate_gscore(W)
    hscores[it] = calculate_hscore(W)
    iscores[it] = calculate_gscore(O)
    total_scores[it] = fscores[it] + lam * gscores[it] + gamma * hscores[it] + beta * iscores[it]
    if verbose:
        print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]}, h={hscores[it]}, i={iscores[it]}, total={total_scores[it]}')


def ini_sonnmf(itermax):
    fscores = np.empty((itermax + 1,))
    gscores = np.empty((itermax + 1,))
    hscores = np.empty((itermax + 1,))
    iscores = np.empty((itermax + 1,))
    total_scores = np.empty((itermax + 1,))
    return fscores, gscores, hscores, iscores, total_scores
