import time
from sonnmf.old.utils import ini_sonnmf, calculate_scores_and_report
from sonnmf.old.update_H import base as base_H_func, precomputed_vars_and_nesterov_acc
from sonnmf.old.update_W import base as base_W
EARLY_STOP_TOL = 1e-6


def sonnmf(M, W, H, lam=0.0, w_update_method='proximal_averaging', itermin=500, itermax=10000, max_minutes=60, H_update_iters=1,
           W_update_iters=10, accelerate_H_update=False, early_stop=True, verbose=False):
    
    start_time = time.time()
    fscores, gscores, total_scores = ini_sonnmf(itermax)

    it = 0
    calculate_scores_and_report(H, M, W, fscores, gscores, it, lam, total_scores, verbose)

    for it in range(1, itermax + 1):
        # update H
        if accelerate_H_update:
            H = precomputed_vars_and_nesterov_acc(H, H_update_iters, M, W)
        else:
            H = base_H_func(H, H_update_iters, M, W)

        # update W
        W = base_W(w_update_method, H, M, W, W_update_iters, lam)

        calculate_scores_and_report(H, M, W, fscores, gscores, it, lam, total_scores, verbose)

        if early_stop and it > itermin:
            if abs(total_scores[it] - total_scores[it - 1]) / total_scores[it - 1] < EARLY_STOP_TOL:
                print(f'Early stopping condition reached at iteration {it}.')
                break
        if time.time() - start_time > max_minutes * 60:
            print(f'Time limit ({max_minutes} minutes) reached at iteration {it}.')
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], total_scores[:it + 1]
