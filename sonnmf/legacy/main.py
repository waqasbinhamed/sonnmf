import time
from sonnmf.legacy.utils import ini_sonnmf, calculate_scores_and_report
from sonnmf.legacy.update_H import update_h_basic, update_h_nesterov
from sonnmf.legacy.update_W import update_w

EARLY_STOP_TOLERANCE = 1e-6

def sonnmf(M, W, H, lam=0.0, w_update_method='proximal_averaging', itermin=200, itermax=1000, max_minutes=60, H_update_iters=1,
           W_update_iters=10, accelerate_H_update=False, early_stop=True, verbose=False):
    """
    Performs non-negative matrix factorization (NMF) with penalties on W and H.

    Reference: Algorithm 4 in [arXiv:2307.00706]

    Parameters:
        M (np.ndarray): The matrix to be factorized.
        W (np.ndarray): The initial matrix for W.
        H (np.ndarray): The initial matrix for H.
        lam (float): Penalty parameter for the group sparsity term.
        w_update_method (str): The method to use for updating W ('proximal_averaging', 'admm', etc.).
        itermin (int): Minimum number of iterations to perform.
        itermax (int): Maximum number of iterations to perform.
        max_minutes (int): Maximum runtime in minutes.
        H_update_iters (int): Number of iterations for updating H.
        W_update_iters (int): Number of iterations for updating W.
        accelerate_H_update (bool): Whether to use Nesterov acceleration for H updates.
        early_stop (bool): Whether to use early stopping based on the change in objective function value.
        verbose (bool): Whether to print progress updates.

    Returns:
        W (np.ndarray): The learned matrix W.
        H (np.ndarray): The learned matrix H.
        fscores (np.ndarray): Array of the objective function value at each iteration.
        gscores (np.ndarray): Array of the group sparsity term value at each iteration.
        total_scores (np.ndarray): Array of the total objective function value at each iteration.
    """
    start_time = time.time()

    # Initialize score arrays
    fscores, gscores, total_scores = ini_sonnmf(itermax)

    # Initial score calculation
    it = 0
    calculate_scores_and_report(H, M, W, fscores, gscores, it, lam, total_scores, verbose)

    for it in range(1, itermax + 1):
        # Update H
        if accelerate_H_update:
            H = update_h_nesterov(H, H_update_iters, M, W)
        else:
            H = update_h_basic(H, H_update_iters, M, W)

        # Update W
        W = update_w(w_update_method, H, M, W, W_update_iters, lam)

        # Calculate and report scores
        calculate_scores_and_report(H, M, W, fscores, gscores, it, lam, total_scores, verbose)

        # Check early stopping condition
        if early_stop and it > itermin:
            relative_change = abs(total_scores[it] - total_scores[it - 1]) / total_scores[it - 1]
            if relative_change < EARLY_STOP_TOLERANCE:
                print(f'Early stopping condition reached at iteration {it}.')
                break

        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > max_minutes * 60:
            print(f'Time limit ({max_minutes} minutes) reached at iteration {it}.')
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], total_scores[:it + 1]
